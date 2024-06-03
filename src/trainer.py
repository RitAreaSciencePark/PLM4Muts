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

def ddp_setup():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def print_peak_memory(prefix, device):
    ma  = torch.cuda.memory_allocated(device)
    mma = torch.cuda.max_memory_allocated(device)
    mmr = torch.cuda.max_memory_reserved(device)
    tot = torch.cuda.get_device_properties(0).total_memory
    print(f"{prefix} - device={device}: allocated=[{ma//1e6} MB]\tmax allocated=[{mma//1e6} MB]\treserved=[{mmr//1e6} MB]\ttotal=[{tot//1e6} MB]", flush=True)

class Trainer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_fn: torch.nn.functional, 
        output_dir: str,
        max_epochs: int,
        seeds: tuple
    ) -> None:
        self.max_epochs  = max_epochs
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.loss_fn     = loss_fn
        self.output_dir  = output_dir
        self.local_rank  = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.epochs_run  = 0
        self.seeds       = seeds

    def _load_snapshot(self, model, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        model_name = snapshot["MODEL_NAME"]
        print(f"Resuming the model {model_name} training from snapshot at Epoch {self.epochs_run+1}", flush=True)

    def _save_snapshot(self, filename, epoch, save_onnx):
        snapshot_file = self.snapshot_dir + filename + ".pt"
        snapshot = {"MODEL_STATE": self.model.module.state_dict(),"EPOCHS_RUN": epoch+1,"MODEL_NAME":self.model.module.name}
        torch.save(snapshot, snapshot_file)
        print(f"Epoch {epoch+1} | Training snapshot saved at {snapshot_file}")
        if save_onnx==True:
            onnx_file = self.snapshot_dir +"/onnx"+ filename +".onnx"
            (first_cpu,second_cpu,pos_cpu), (input_names,output_names,dynamic_axes) = self.model.module.onnx_model_args()
            first_gpu  =  first_cpu.to(self.local_rank)
            second_gpu = second_cpu.to(self.local_rank)
            pos_gpu    =    pos_cpu.to(self.local_rank)
            torch.onnx.export(self.model.module, (first_gpu, second_gpu, pos_gpu), onnx_file, 
                              export_params=True, opset_version=14, do_constant_folding=True,
                              input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
            print(f"Epoch {epoch+1} | Training snapshot saved at {onnx_file}")
            del first_gpu 
            del second_gpu
            del pos_gpu

    def initialize_files(self):
        self.result_dir    =  self.output_dir + "/results"
        self.snapshot_dir  =  self.output_dir + "/snapshots"
        self.train_logfile =  self.result_dir + f"/train_{self.train_dl.name}_metrics.log"
        self.val_logfile   =  self.result_dir + f"/val_{self.val_dl.name}_metrics.log"
        self.test_logfiles = [self.result_dir + f"/test_{test.name}_metrics.log" for test in self.test_dls] 
        self.seeds_file    =  self.result_dir + "/seeds.log"
        self.early_file    =  self.result_dir + "/early_stopping_epoch.log"
        if self.global_rank == 0:
            if not(os.path.exists(self.output_dir) and os.path.isdir(self.output_dir)):
                os.makedirs(self.output_dir)
            if not(os.path.exists(self.result_dir) and os.path.isdir(self.result_dir)):
                os.makedirs(self.result_dir)
            if not(os.path.exists(self.snapshot_dir) and os.path.isdir(self.snapshot_dir)):
                os.makedirs(self.snapshot_dir)            
                os.makedirs(self.snapshot_dir + "/onnx")            
            with open(self.seeds_file, "w") as seeds_f:
                seeds_f.write(f"seeds = {self.seeds}\n")
            with open(self.train_logfile, "w") as tr_log:
                tr_log.write("epoch,rmse,mae,corr\n")
            with open(self.val_logfile, "w") as v_log:
                v_log.write("epoch,rmse,mae,corr\n")
            for test_logfile in self.test_logfiles:
                with open(test_logfile, "w") as t_log:
                    t_log.write("epoch,rmse,mae,corr\n")

    def train_batch(self,  batch):
        X, Y, code = batch
        code = "".join(code)
        first_cpu, second_cpu, pos_cpu = self.model.module.preprocess(*X)
        first_gpu  =  first_cpu.to(self.local_rank)
        second_gpu = second_cpu.to(self.local_rank)
        pos_gpu    =    pos_cpu.to(self.local_rank)
        Y_gpu    = Y.to(self.local_rank)
        #print_peak_memory(f"DEBUG_A: code={code}, first=({first_gpu.shape},{first_gpu.device}) second=({second_gpu.shape},{second_gpu.device}), Y_gpu=({Y_gpu.shape},{Y_gpu.device})",self.local_rank)
        self.optimizer.zero_grad()
        with autocast(dtype=torch.float16):
            Yhat_gpu = self.model(first_gpu, second_gpu, pos_gpu)
            loss     = self.loss_fn(Yhat_gpu, Y_gpu)
            #print_peak_memory(f"DEBUG_B: code={code}, first=({first_gpu.shape},{first_gpu.device}) second=({second_gpu.shape},{second_gpu.device}), Y_gpu=({Y_gpu.shape},{Y_gpu.device}), Yhat_gpu=({Yhat_gpu.shape},{Yhat_gpu.device}), loss={loss}", self.local_rank)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), max_norm = 10.)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        Yhat_cpu = Yhat_gpu.detach().to("cpu").item()
        Y_cpu    =    Y_gpu.detach().to("cpu").item()
        del first_gpu
        del second_gpu
        del pos_gpu
        del Y_gpu  
        del Yhat_gpu
        #torch.cuda.empty_cache()
        return Yhat_cpu, Y_cpu

    def all_gather_lab_preds(self, preds, labels):
        size       = dist.get_world_size()
        preds_gpu  = preds.to(self.local_rank)
        labels_gpu = labels.to(self.local_rank)
        prediction_list_gpu = [torch.zeros_like(preds).to(self.local_rank)  for _ in range(size)]
        labels_list_gpu     = [torch.zeros_like(labels).to(self.local_rank) for _ in range(size)]
        dist.all_gather(prediction_list_gpu, preds_gpu)
        dist.all_gather(labels_list_gpu, labels_gpu)
        
        global_preds_cpu  = torch.tensor([], dtype=torch.float32).to("cpu")
        global_labels_cpu = torch.tensor([], dtype=torch.float32).to("cpu")
        for t1_gpu in prediction_list_gpu:
            t1_cpu=t1_gpu.to("cpu")
            global_preds_cpu = torch.cat((global_preds_cpu,t1_cpu), dim=0)
        for t2_gpu in labels_list_gpu:
            t2_cpu=t2_gpu.to("cpu")
            global_labels_cpu = torch.cat((global_labels_cpu,t2_cpu),dim=0)
        
        for t1_gpu in prediction_list_gpu:
            del t1_gpu
        for t2_gpu in labels_list_gpu:
            del t2_gpu
        del prediction_list_gpu
        del labels_list_gpu
        del preds_gpu
        del labels_gpu
        
        return global_preds_cpu, global_labels_cpu

    def train_epoch(self, epoch, train_proteindataloader):
        self.scaler = torch.cuda.amp.GradScaler()
        len_dataloader = len(train_proteindataloader.dataloader)
        local_t_preds, local_t_labels = torch.zeros(len_dataloader), torch.zeros(len_dataloader)
        self.model.train()
        train_proteindataloader.dataloader.sampler.set_epoch(epoch)
        for idx, batch in enumerate(train_proteindataloader.dataloader):
            #print(f"({self.local_rank}) idx={idx} / {len_dataloader}", flush=True)
            Yhat, Y = self.train_batch(batch)
            local_t_preds[idx] = Yhat
            local_t_labels[idx] = Y
        global_t_preds, global_t_labels = self.all_gather_lab_preds(local_t_preds, local_t_labels)
        #print(f"DEBUG: global_t_preds=({global_t_preds.shape},{global_t_preds.device})\nglobal_t_labels=({global_t_labels.shape},{global_t_labels.device})\nlocal_t_preds=({local_t_preds.shape},{local_t_preds.device})\tlocal_t_labels=({local_t_labels.shape},{local_t_labels.device})", flush=True)
        return global_t_labels, global_t_preds, local_t_labels, local_t_preds

    def valid_batch(self, batch):
        X, Y, _ = batch
        first_cpu, second_cpu, pos_cpu = self.model.module.preprocess(*X)
        first_gpu  =  first_cpu.to(self.local_rank)
        second_gpu = second_cpu.to(self.local_rank)
        pos_gpu    =    pos_cpu.to(self.local_rank)
        Yhat_gpu = self.model(first_gpu, second_gpu, pos_gpu)
        Yhat_cpu = Yhat_gpu.detach().to("cpu").item()
        Y_cpu    =        Y.detach().to("cpu").item()
        del first_gpu
        del second_gpu
        del pos_gpu
        del Yhat_gpu 
        #torch.cuda.empty_cache()
        return Yhat_cpu, Y_cpu

    def valid_epoch(self, epoch, val_proteindataloader):
        self.model.eval()
        len_dataloader = len(val_proteindataloader.dataloader)
        local_v_preds, local_v_labels = torch.zeros(len_dataloader), torch.zeros(len_dataloader)
        with torch.no_grad():
            for idx, batch in enumerate(val_proteindataloader.dataloader):
                Yhat, Y = self.valid_batch(batch)
                local_v_preds[idx] = Yhat
                local_v_labels[idx]= Y
        global_v_preds, global_v_labels = self.all_gather_lab_preds(local_v_preds, local_v_labels)
        return global_v_labels, global_v_preds, local_v_labels, local_v_preds

    def train(self, model, train_dl, val_dl, test_dls):
        print(f"I am rank {self.local_rank}", flush=True)
        self.model = model
        self.model_name = self.model.module.name
        #self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.train_dl = train_dl
        self.val_dl   =  val_dl
        self.test_dls = test_dls
        self.initialize_files()
        self.train_rmse   = torch.zeros(self.max_epochs)
        self.train_mae    = torch.zeros(self.max_epochs)
        self.train_corr   = torch.zeros(self.max_epochs)
        self.val_rmse     = torch.zeros(self.max_epochs)
        self.val_mae      = torch.zeros(self.max_epochs)
        self.val_corr     = torch.zeros(self.max_epochs)
        self.test_rmses   = torch.zeros(len(self.test_dls),self.max_epochs)
        self.test_maes    = torch.zeros(len(self.test_dls),self.max_epochs)
        self.test_corrs   = torch.zeros(len(self.test_dls),self.max_epochs)
        # Early Stopping
        patience  = 5
        min_delta = 0.005
        counter   = 0
        min_validation_loss = float('inf')
        self.stopped_epoch  = self.max_epochs
        self.best_epoch     = self.max_epochs 
        # Start epoch loop
        print_peak_memory("Start finetuning", self.local_rank)
        for epoch in range(self.epochs_run, self.max_epochs):
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
            print(f"{self.train_dl.name}\tGPU:{self.global_rank}\tepoch:{epoch+1}/{self.max_epochs}\t"
                  f"rmse = {g_t_rmse} ({l_t_rmse})\t"
                  f"mae = {g_t_mae} ({l_t_mae})\t"
                  f"corr = {g_t_corr} ({l_t_corr})", flush=True)
            
            dist.barrier()
            #print_peak_memory(f"Epoch {epoch+1}: train completed", self.local_rank)
            dist.barrier()
            if self.global_rank == 0:
                print(f"Validation ongoing on MAE for {self.val_dl.name}", flush=True)
                self._save_snapshot("/checkpoint", epoch, save_onnx=False)
            dist.barrier()

            g_v_labels, g_v_preds, l_v_labels, l_v_preds = self.valid_epoch(epoch, self.val_dl)
            g_v_mse  = torch.mean(         (g_v_labels - g_v_preds)**2)
            g_v_mae  = torch.mean(torch.abs(g_v_labels - g_v_preds)   )
            g_v_rmse = torch.sqrt(g_v_mse)
            g_v_corr, _ = pearsonr(g_v_labels.tolist(), g_v_preds.tolist())
            l_v_mse  = torch.mean(         (l_v_labels - l_v_preds)**2)
            l_v_mae  = torch.mean(torch.abs(l_v_labels - l_v_preds)   )
            l_v_rmse = torch.sqrt(l_v_mse)
            l_v_corr, _ = pearsonr(l_v_labels.tolist(), l_v_preds.tolist())
            self.val_mae[epoch]  = g_v_mae
            self.val_corr[epoch] = g_v_corr
            self.val_rmse[epoch] = g_v_rmse
            print(f"{self.val_dl.name}\tGPU:{self.global_rank}\tepoch:{epoch+1}/{self.max_epochs}\t"
                      f"rmse = {self.val_rmse[epoch]} ({l_v_rmse})\t"
                      f"mae = {self.val_mae[epoch]} ({l_v_mae})\t"
                      f"corr = {self.val_corr[epoch]} ({l_v_corr})", flush=True)

            dist.barrier()
            if self.val_mae[epoch] < min_validation_loss: 
                if self.global_rank == 0:
                    print(f"Early Stopping counter set to 0 at epoch {epoch+1}: Cur_Val_MAE={self.val_mae[epoch]}<{min_validation_loss}=Prev_Val_MAE", flush=True)
                    self._save_snapshot(f"/{self.model_name}", epoch, save_onnx=True)
                min_validation_loss = self.val_mae[epoch] 
                self.best_epoch = epoch + 1
                counter = 0
            
            dist.barrier()
            #print_peak_memory(f"Epoch {epoch+1}: validation completed", self.local_rank)
            dist.barrier()

            for test_no, test_dl in enumerate(self.test_dls):
                g_t_labels, g_t_preds, l_t_labels, l_t_preds = self.valid_epoch(epoch, test_dl)
                g_t_mse  = torch.mean(         (g_t_labels - g_t_preds)**2)
                g_t_mae  = torch.mean(torch.abs(g_t_labels - g_t_preds)   )
                g_t_rmse =  torch.sqrt(g_t_mse)
                g_t_corr, _ = pearsonr(g_t_labels.tolist(),  g_t_preds.tolist())
                l_t_mse  = torch.mean(         (l_t_labels - l_t_preds)**2)
                l_t_mae  = torch.mean(torch.abs(l_t_labels - l_t_preds)   )
                l_t_rmse = torch.sqrt(l_t_mse)
                l_t_corr, _ = pearsonr(l_t_labels.tolist(), l_t_preds.tolist())
                self.test_maes[ test_no,epoch] = g_t_mae
                self.test_corrs[test_no,epoch] = g_t_corr
                self.test_rmses[test_no,epoch] = g_t_rmse
                print(f"{test_dl.name}\tGPU:{self.global_rank}\tepoch:{epoch+1}/{self.max_epochs}\t"
                      f"rmse = {self.test_rmses[test_no,epoch]} ({l_t_rmse})\t"
                      f"mae = {self.test_maes[test_no,epoch]} ({l_t_mae})\t"
                      f"corr = {self.test_corrs[test_no,epoch]} ({l_t_corr})", flush=True)
            #print_peak_memory(f"Epoch {epoch+1}: test completed", self.local_rank)

            dist.barrier()
            if self.val_mae[epoch] > (min_validation_loss + min_delta):
                counter += 1
                if self.global_rank == 0:
                    print(f"Early Stopping counter increased at epoch {epoch+1} to {counter}: Cur_Val_MAE={self.val_mae[epoch]}>{min_validation_loss + min_delta}=Prev_Val_MAE ({min_validation_loss}) + min_delta ({min_delta})", flush=True)
            
            dist.barrier()
            #print_peak_memory(f"Epoch {epoch+1}: epoch completed", self.local_rank)
            if counter >= patience:
                if self.global_rank == 0:
                    print(f"Early Stopping at epoch {epoch + 1}: counter={counter} >= {patience}=patience", flush=True)
                self.stopped_epoch = epoch + 1
                dist.barrier()
                break
            else:
                if self.global_rank == 0:
                    print(f"Early Stopping counter at epoch {epoch+1}: {counter}/{patience}", flush=True)
                dist.barrier()
        
        print_peak_memory(f"Finetuning completed", self.local_rank)
        dist.barrier()
        if self.global_rank == 0:
            CHECKPOINT_PATH = self.snapshot_dir + "/checkpoint.pt"
            os.remove(CHECKPOINT_PATH)
            with open(self.early_file, "w") as early_log:
                early_log.write(f"Saved epoch (best MAE on validation set): {self.best_epoch}\n")
            for epoch in range(0, self.stopped_epoch):
                with open(self.train_logfile, "a") as tr_log:
                    tr_log.write(f"{epoch+1},{self.train_rmse[epoch]},{self.train_mae[epoch]},{self.train_corr[epoch]}\n")
                with open(self.val_logfile, "a") as v_log:
                    v_log.write(f"{epoch+1},{self.val_rmse[epoch]},{self.val_mae[epoch]},{self.val_corr[epoch]}\n")
                for test_no, test_dl in enumerate(self.test_dls):
                    with open(self.test_logfiles[test_no], "a") as t_log:
                        t_log.write(f"{epoch+1},{self.test_rmses[test_no,epoch]},{self.test_maes[test_no,epoch]},{self.test_corrs[test_no,epoch]}\n")
        dist.barrier()
        

    def plot(self, dataframe, train_name, val_name, test_names, ylabel, title):
        colors_test = cm.viridis(torch.linspace(0, 1, len(test_names)))
        fig, ax = plt.subplots(ncols=1, figsize=(8, 6))
        plt.plot(dataframe["epoch"], dataframe[train_name],  label=train_name, color="black", linestyle=":")
        plt.plot(dataframe["epoch"], dataframe[val_name],    label=val_name,   color="grey",  linestyle="-.")
        for i, test_name in enumerate(test_names):
            plt.plot(dataframe["epoch"], dataframe[test_name],   label=test_name,  color=colors_test[i], linestyle="-")
        ax.axvline(self.best_epoch, color='red', ls=':')
        ax.text(self.best_epoch,0.99,'Saved epoch',color='red',ha='right',va='top',rotation=90,transform=ax.get_xaxis_transform())
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig( self.result_dir + f"/epochs_{ylabel}.png", bbox_inches = 'tight')
        plt.clf()

    def describe(self):
        dist.barrier()
        if self.global_rank == 0:
            test_rmse_names  = [ "test_" + s.name + "_rmse" for s in self.test_dls]
            test_mae_names   = [ "test_" + s.name + "_mae"  for s in self.test_dls]
            test_corr_names  = [ "test_" + s.name + "_corr" for s in self.test_dls]
            
            val_rmse_name  =  "val_" + self.val_dl.name + "_rmse" 
            val_mae_name   =  "val_" + self.val_dl.name + "_mae"  
            val_corr_name  =  "val_" + self.val_dl.name + "_corr" 
            val_rmse_dict = { val_rmse_name: self.val_rmse}
            val_mae_dict  = { val_mae_name:  self.val_mae}
            val_corr_dict = { val_corr_name: self.val_corr}

            train_rmse_name = "train_" + self.train_dl.name + '_rmse'
            train_mae_name  = "train_" + self.train_dl.name + '_mae'
            train_corr_name = "train_" + self.train_dl.name + '_corr'
            train_rmse_dict = { train_rmse_name: self.train_rmse}
            train_mae_dict  = { train_mae_name:  self.train_mae}
            train_corr_dict = { train_corr_name: self.train_corr}

            test_rmse_df   = pd.DataFrame.from_dict(dict(zip(test_rmse_names, self.test_rmses)))
            test_mae_df    = pd.DataFrame.from_dict(dict(zip(test_mae_names,  self.test_maes)))
            test_corr_df   = pd.DataFrame.from_dict(dict(zip(test_corr_names, self.test_corrs)))
            
            val_rmse_df  = pd.DataFrame.from_dict(val_rmse_dict)
            val_mae_df   = pd.DataFrame.from_dict(val_mae_dict)
            val_corr_df  = pd.DataFrame.from_dict(val_corr_dict)

            train_rmse_df  = pd.DataFrame.from_dict(train_rmse_dict)
            train_mae_df   = pd.DataFrame.from_dict(train_mae_dict)
            train_corr_df  = pd.DataFrame.from_dict(train_corr_dict)
             
            train_rmse_df["epoch"]  = train_rmse_df.index + 1
            train_mae_df["epoch"]   = train_mae_df.index  + 1
            train_corr_df["epoch"]  = train_corr_df.index + 1
            val_corr_df["epoch"]    = val_corr_df.index   + 1
            val_mae_df["epoch"]     = val_mae_df.index    + 1
            val_rmse_df["epoch"]    = val_rmse_df.index   + 1
            test_corr_df["epoch"]   = test_corr_df.index  + 1
            test_mae_df["epoch"]    = test_mae_df.index   + 1
            test_rmse_df["epoch"]   = test_rmse_df.index  + 1

            df = pd.concat([frame.set_index("epoch") for frame in [train_rmse_df, train_mae_df, train_corr_df, val_rmse_df, val_mae_df, val_corr_df, test_rmse_df, test_mae_df, test_corr_df]],
               axis=1, join="inner").reset_index()

            df = df[:self.stopped_epoch]
            print(df, flush=True)

            df.to_csv( self.result_dir + f"/epochs_statistics.csv", index=False)
            self.plot(df, train_rmse_name, val_rmse_name, test_rmse_names, ylabel="rsme", title=f"Model: {self.model_name}")
            self.plot(df, train_mae_name,  val_mae_name,  test_mae_names,  ylabel="mae",  title=f"Model: {self.model_name}")
            self.plot(df, train_corr_name, val_corr_name, test_corr_names, ylabel="corr", title=f"Model: {self.model_name}")
        dist.barrier()


