import argparse
from Bio import SeqIO
import csv
import esm
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import random
import re
import scipy
from scipy import stats 
from scipy.stats import pearsonr
import string
import time
import torch
import torch.distributed  as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from transformers import T5Tokenizer
from typing import List, Tuple
import warnings

torch.cuda.empty_cache()
#warnings.filterwarnings("ignore")



#data-preprocessing step
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def print_peak_memory(prefix, device):
    mma = torch.cuda.max_memory_allocated(device)
    mmr = torch.cuda.max_memory_reserved(device)
    tot = torch.cuda.get_device_properties(0).total_memory
    print(f"{prefix}: allocated [{mma//1e6} MB]\treserved [{mmr//1e6} MB]\ttotal [{tot//1e6} MB]")


class Trainer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        save_every: int,
        loss_fn: torch.nn.functional, 
        output_dir: str,
        max_epochs: int,
    ) -> None:
        self.max_epochs  = max_epochs
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.save_every  = save_every
        self.loss_fn     = loss_fn
        self.output_dir  = output_dir
        self.save_every  = save_every
        self.local_rank  = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.epochs_run  = 0

    def _load_snapshot(self, model, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        model.module.load_state_dict(snapshot["MODEL_STATE"])
        epoch = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {epoch}")

    def _save_snapshot(self, epoch):
        print("AAAA", self.model.module.state_dict())
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def initialize_files(self):
        self.result_dir    = self.output_dir + "/results"
        self.snapshot_dir  = self.output_dir + "/snapshot"
        if self.global_rank == 0:
            if not(os.path.exists(self.output_dir) and os.path.isdir(self.output_dir)):
                os.makedirs(self.output_dir)
            if not(os.path.exists(self.result_dir) and os.path.isdir(self.result_dir)):
                os.makedirs(self.result_dir)
            if not(os.path.exists(self.snapshot_dir) and os.path.isdir(self.snapshot_dir)):
                os.makedirs(self.snapshot_dir)            

        self.snapshot_path = self.snapshot_dir + "/snapshot.pt"
        self.train_logfile =  self.result_dir + "/train_metrics.log"
        self.val_logfiles  = [self.result_dir + f"/{val.name}_metrics.log" for val in self.val_dls] 
        
        if self.global_rank == 0:
            with open(self.train_logfile, "w") as t_log:
                t_log.write("epoch,rmse,mae,corr\n")
            for val_logfile in self.val_logfiles:
                with open(val_logfile, "w") as v_log:
                    v_log.write("epoch,rmse,mae,corr\n")
        

    def train_batch(self,  batch):
        X, Y, _ = batch
        #X = [x.to(self.local_rank) for x in X]
        Y = Y.to(self.local_rank)
        with autocast(dtype=torch.bfloat16):
            Y_hat = self.model(*X, self.local_rank).to(self.local_rank)
            loss   = self.loss_fn(Y_hat, Y)
        torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), max_norm = 0.1)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.scaler.step(self.optimizer)
        self.scheduler.step()
        self.scaler.update()
        self.t_labels.extend(Y.cpu().detach())
        self.t_preds.extend(Y_hat.cpu().detach())

    def all_gather_lab_preds(self, preds, labels):
        #size is the world size
        size = dist.get_world_size()
        preds = torch.tensor(preds).to(self.local_rank)
        labels = torch.tensor(labels).to(self.local_rank)
        prediction_list = [torch.zeros_like(preds).to(self.local_rank)  for _ in range(size)]
        labels_list     = [torch.zeros_like(labels).to(self.local_rank) for _ in range(size)]
        dist.all_gather(prediction_list, preds)
        dist.all_gather(labels_list, labels)
        global_preds  = torch.tensor([], dtype=torch.float32).to(self.local_rank)
        global_labels = torch.tensor([], dtype=torch.float32).to(self.local_rank)
        for t1 in prediction_list:
            global_preds = torch.cat((global_preds,t1), dim=0)
        for t2 in labels_list:
            global_labels = torch.cat((global_labels,t2),dim=0)
        return global_preds, global_labels 

    def train_epoch(self, epoch, train_proteindataloader):
        self.scaler = torch.cuda.amp.GradScaler()
        self.t_preds, self.t_labels = [], []
        self.model.train()
        len_dataloader = len(train_proteindataloader.dataloader)
        train_proteindataloader.dataloader.sampler.set_epoch(epoch)
        for idx, batch in enumerate(train_proteindataloader.dataloader):
            #print_peak_memory("Max GPU memory", self.local_rank)
            print(f"{train_proteindataloader.name}\ton GPU {self.global_rank} epoch:{epoch+1}/{self.max_epochs}\tbatch_idx:{idx+1}/{len_dataloader} - {batch[-1]}")
            self.train_batch(batch)
            dist.barrier()
        
        global_t_preds, global_t_labels = self.all_gather_lab_preds(self.t_preds, self.t_labels)
        g_t_labels = global_t_labels.to("cpu")
        g_t_preds  = global_t_preds.to("cpu")
        l_t_labels = torch.tensor(self.t_labels).to("cpu")
        l_t_preds  = torch.tensor(self.t_preds).to("cpu")
        return g_t_labels, g_t_preds, l_t_labels, l_t_preds

    def valid_batch(self, batch):
        X, Y, _ = batch
        #X = [x.to(self.local_rank) for x in X]
        Yhat = self.model(*X, self.local_rank).to(self.local_rank)
        Y_cpu = Y.cpu().detach()
        Yhat_cpu = Yhat.cpu().detach()
        self.v_labels.extend(Y_cpu)
        self.v_preds.extend(Yhat_cpu)

    def valid_epoch(self, epoch, val_proteindataloader):
        self.model.eval()
        self.v_preds, self.v_labels = [], []
        len_dataloader = len(val_proteindataloader.dataloader)
        with torch.no_grad():
            for idx, batch in enumerate(val_proteindataloader.dataloader):
                print(f"{val_proteindataloader.name}\ton GPU {self.global_rank} epoch:{epoch+1}/{self.max_epochs}\tbatch_idx:{idx+1}/{len_dataloader} - {batch[-1]}")
                self.valid_batch(batch)
        global_v_preds, global_v_labels = self.all_gather_lab_preds(self.v_preds, self.v_labels)
        g_v_labels = global_v_labels.to("cpu")
        g_v_preds  = global_v_preds.to("cpu")
        l_v_labels = torch.tensor(self.v_labels).to("cpu")
        l_v_preds  = torch.tensor(self.v_preds).to("cpu")
        return g_v_labels, g_v_preds, l_v_labels, l_v_preds

    def difference_labels_preds(self, cutoff):
        for val_no, val_dl in enumerate(self.val_dls):
            with open(self.result_dir + f"/{val_dl.name}_labels_preds.{self.global_rank}.diffs", "a") as v_diffs:
                v_diffs.write(f"code,pos,ddg,pred\n")
            self.model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(val_dl.dataloader):
                    X, Y, code = batch
                    #X = [x.to(self.local_rank) for x in X]
                    Yhat = self.model(*X, self.local_rank).to(self.local_rank)
                    Y_cpu = Y.cpu().detach().item()
                    Yhat_cpu = Yhat.cpu().detach().item()
                    pos = X[-1]
                    pos_cpu = pos.cpu().detach().item()
                    with open(self.result_dir + f"/{val_dl.name}_labels_preds.{self.global_rank}.diffs", "a") as v_diffs:
                       v_diffs.write(f"{code},{pos_cpu},{Y_cpu},{Yhat_cpu}\n")

    def train(self, model, train_dl, val_dls):
        print(f"I am rank {self.local_rank}")
        self.model_name = model.name
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
#        if os.path.exists(self.snapshot_path):
#            print("Loading snapshot")
#            self._load_snapshot(self.snapshot_path)
        self.train_dl = train_dl
        self.val_dls = val_dls
        self.initialize_files()
        self.train_rmse  = torch.zeros(self.max_epochs)
        self.train_mae   = torch.zeros(self.max_epochs)
        self.train_corr  = torch.zeros(self.max_epochs)
        self.val_rmses   = torch.zeros(len(self.val_dls),self.max_epochs)
        self.val_maes    = torch.zeros(len(self.val_dls),self.max_epochs)
        self.val_corrs   = torch.zeros(len(self.val_dls),self.max_epochs)
        old_corr = 0
        for epoch in range(self.epochs_run, self.max_epochs):
            dist.barrier()
            g_t_labels, g_t_preds, l_t_labels, l_t_preds = self.train_epoch(epoch, self.train_dl)
            g_t_mse  = torch.mean(         (g_t_labels - g_t_preds)**2)
            g_t_mae  = torch.mean(torch.abs(g_t_labels - g_t_preds)   )
            g_t_rmse = torch.sqrt(g_t_mse)
            g_t_corr, _ = pearsonr(g_t_labels.tolist(), g_t_preds.tolist())
            l_t_mse  = torch.mean(         (l_t_labels - l_t_preds)**2)
            l_t_mae  = torch.mean(torch.abs(l_t_labels - l_t_preds)   )
            l_t_rmse = torch.sqrt(l_t_mse)
            l_t_corr, _ = pearsonr(l_t_labels.tolist(), l_t_preds.tolist())
            self.train_mae[epoch]  = g_t_mae
            self.train_corr[epoch] = g_t_corr
            self.train_rmse[epoch] = g_t_rmse
            print(f"{self.train_dl.name}: on GPU {self.global_rank} epoch {epoch+1}/{self.max_epochs}\t"
                  f"rmse = {g_t_rmse} / {l_t_rmse}\t"
                  f"mae = {g_t_mae} / {l_t_mae}\t"
                  f"corr = {g_t_corr} / {l_t_corr}")
            if self.local_rank == 0:
                with open(self.train_logfile, "a") as t_log:
                    t_log.write(f"{epoch+1},{self.train_rmse[epoch]},{self.train_mae[epoch]},{self.train_corr[epoch]}\n")
            
            #if self.local_rank == 0 and epoch % self.save_every == 0:
            #    self._save_snapshot(epoch)
            
            for val_no, val_dl in enumerate(self.val_dls):
                g_v_labels, g_v_preds, l_v_labels, l_v_preds = self.valid_epoch(epoch, val_dl)
                g_v_mse  = torch.mean(         (g_v_labels - g_v_preds)**2)
                g_v_mae  = torch.mean(torch.abs(g_v_labels - g_v_preds)   )
                g_v_rmse = torch.sqrt(g_v_mse)
                g_v_corr, _ = pearsonr(g_v_labels.tolist(), g_v_preds.tolist())
                l_v_mse  = torch.mean(         (l_v_labels - l_v_preds)**2)
                l_v_mae  = torch.mean(torch.abs(l_v_labels - l_v_preds)   )
                l_v_rmse = torch.sqrt(l_v_mse)
                l_v_corr, _ = pearsonr(l_v_labels.tolist(), l_v_preds.tolist())
                self.val_maes[val_no,epoch]  = g_v_mae
                self.val_corrs[val_no,epoch] = g_v_corr
                self.val_rmses[val_no,epoch] = g_v_rmse
                print(f"{val_dl.name}: on GPU {self.global_rank} epoch {epoch+1}/{self.max_epochs}\t"
                      f"rmse = {self.val_rmses[val_no,epoch]} / {l_v_rmse}\t"
                      f"mae = {self.val_maes[val_no,epoch]} / {l_v_mae}\t"
                      f"corr = {self.val_corrs[val_no,epoch]} / {l_v_corr}")

                dist.barrier()
                if self.local_rank == 0 and self.val_corrs[0,epoch] > old_corr:
                    self._save_snapshot(epoch)
                    old_corr = self.val_corrs[0,epoch] 
                dist.barrier()
                if self.local_rank == 0:
                    with open(self.result_dir + f"/{val_dl.name}_metrics.log", "a") as v_log:
                        v_log.write(f"{epoch+1},{self.val_rmses[val_no,epoch]},{self.val_maes[val_no,epoch]},{self.val_corrs[val_no,epoch]}\n")
        dist.barrier()
        self.difference_labels_preds(cutoff=0.0)
        dist.barrier()
        #if self.global_rank == 0:
        #    self._save_snapshot(epoch)
        #dist.barrier()
        
    def test(self, test_model, test_dls): 
        self.test_model_name = test_model.name
        self.test_model = test_model.to(self.local_rank)
        self.test_model = DDP(self.test_model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.test_dls = test_dls
        self.test_rmses   = torch.zeros(len(self.test_dls))
        self.test_maes    = torch.zeros(len(self.test_dls))
        self.test_corrs   = torch.zeros(len(self.test_dls))
        if os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(self.test_model, self.snapshot_path)
        
        self.test_model.eval()
        for test_no, test_dl in enumerate(self.test_dls):
            self.t_preds, self.t_labels = [], []
            len_dataloader = len(test_dl.dataloader)
            with torch.no_grad():
                for idx, batch in enumerate(test_dl.dataloader):
                    print(f"{test_dl.name}\ton GPU {self.global_rank} test:\tbatch_idx:{idx+1}/{len_dataloader}-{batch[-1]}")
                    testX, testY, _ = batch
                    testYhat = self.test_model.module.forward(*testX, self.local_rank).to(self.local_rank)
                    testY_cpu = testY.cpu().detach()
                    testYhat_cpu = testYhat.cpu().detach()
                    self.t_labels.extend(testY_cpu)
                    self.t_preds.extend(testYhat_cpu)
            l_t_labels = torch.tensor(self.t_labels).to("cpu")
            l_t_preds  = torch.tensor(self.t_preds).to("cpu")
            l_t_mse  = torch.mean(         (l_t_labels - l_t_preds)**2)
            l_t_mae  = torch.mean(torch.abs(l_t_labels - l_t_preds)   )
            l_t_rmse = torch.sqrt(l_t_mse)
            l_t_corr, _ = pearsonr(l_t_labels.tolist(), l_t_preds.tolist())
            self.test_maes[test_no]  = l_t_mae
            self.test_corrs[test_no] = l_t_corr
            self.test_rmses[test_no] = l_t_rmse
            print(f"{test_dl.name}: on GPU {self.global_rank} test\t"
                      f"rmse = {self.test_rmses[test_no]}\t"
                      f"mae = {self.test_maes[test_no]}\t"
                      f"corr = {self.test_corrs[test_no]}")

            dist.barrier()
            if self.local_rank == 0:
                with open(self.result_dir + f"/{test_dl.name}_{self.global_rank}_test.log", "a") as v_log:
                        v_log.write(f"test,{self.test_rmses[test_no]},{self.test_maes[test_no]},{self.test_corrs[test_no]}\n")

    def plot(self, dataframe, train_name, val_names, ylabel, title):
        colors = cm.rainbow(torch.linspace(0, 1, len(val_names)))
        plt.figure(figsize=(8,6))
        plt.plot(dataframe["epoch"],     dataframe[train_name], label=train_name, color="black",   linestyle="-.")
        for i, val_name in enumerate(val_names):
            plt.plot(dataframe["epoch"], dataframe[val_name],   label=val_name,   color=colors[i], linestyle="-")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig( self.result_dir + f"/epochs_{ylabel}.png")
        plt.clf()


    def describe(self):
        if self.local_rank == 0:
            val_rmse_names  = [ s.name + "_rmse" for s in self.val_dls]
            val_mae_names   = [ s.name + "_mae"  for s in self.val_dls]
            val_corr_names  = [ s.name + "_corr" for s in self.val_dls]
            train_rmse_name = self.train_dl.name + '_rmse'
            train_mae_name  = self.train_dl.name + '_mae'
            train_corr_name = self.train_dl.name + '_corr'
            train_rmse_dict = { train_rmse_name: self.train_rmse}
            train_mae_dict  = { train_mae_name:  self.train_mae}
            train_corr_dict = { train_corr_name: self.train_corr}

            val_rmse_df   = pd.DataFrame.from_dict(dict(zip(val_rmse_names,  self.val_rmses)))
            val_mae_df    = pd.DataFrame.from_dict(dict(zip(val_mae_names,   self.val_maes)))
            val_corr_df   = pd.DataFrame.from_dict(dict(zip(val_corr_names,  self.val_corrs)))
            train_rmse_df = pd.DataFrame.from_dict(train_rmse_dict)
            train_mae_df  = pd.DataFrame.from_dict(train_mae_dict)
            train_corr_df = pd.DataFrame.from_dict(train_corr_dict)
             
            train_rmse_df["epoch"] = train_rmse_df.index
            train_mae_df["epoch"]  = train_mae_df.index
            train_corr_df["epoch"] = train_corr_df.index
            val_corr_df["epoch"]   = val_corr_df.index
            val_mae_df["epoch"]    = val_mae_df.index
            val_rmse_df["epoch"]   = val_rmse_df.index

            df = pd.concat([frame.set_index("epoch") for frame in [train_rmse_df, train_mae_df, train_corr_df, val_rmse_df, val_mae_df, val_corr_df]],
               axis=1, join="inner").reset_index()

            print(df, flush=True)

            df.to_csv( self.result_dir + f"/epochs_statistics_{self.global_rank}.csv")
            self.plot(df, train_rmse_name, val_rmse_names, ylabel="rsme", title="Model: {self.model_name}")
            self.plot(df, train_mae_name,  val_mae_names,  ylabel="mae",  title="Model: {self.model_name}")
            self.plot(df, train_corr_name, val_corr_names, ylabel="corr", title="Model: {self.model_name}")
        dist.barrier()

    def free_memory(self):
        del self.model
        torch.cuda.empty_cache()

