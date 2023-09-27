import math
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import os
import re
import scipy
from scipy import stats
from scipy.stats import pearsonr
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast
from transformers import T5Tokenizer
import warnings

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_world_size, get_rank, all_gather

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def print_peak_memory(prefix, device):
    mma = torch.cuda.max_memory_allocated(device)
    mmr = torch.cuda.max_memory_reserved(device)
    tot = torch.cuda.get_device_properties(0).total_memory
    print(f"{prefix}: allocated [{mma//1e6} MB]\treserved [{mmr//1e6} MB]\ttotal [{tot//1e6} MB]")

def from_cvs_files_in_dir_to_dfs_list(dir_path):
    print(dir_path)
    datasets = os.listdir(dir_path)
    print(datasets)
    #datasets_names = [ s.rsplit('/', 1)[1].rsplit('.', 1)[0]  for s in datasets ]
    datasets_names = [ s.rsplit('.', 1)[0]  for s in datasets ]
    print(datasets_names)
    dfs = [None] * len(datasets)
    for i,d in enumerate(datasets):
        d_path = os.path.join(dir_path, d)
        dfs[i] = pd.read_csv(d_path, sep=',')
    return dfs, datasets_names

### Dataset Definition

class ProteinDataset(Dataset):
    def __init__(self, df, name):
        self.name = name
        self.df = df
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        lengths = [len(s) for s in df['wild_type'].to_list()]
        self.max_length = max(lengths) + 2

    def __getitem__(self, idx):
        wild_seq = self.df.iloc[idx]['wild_type']
        mut_seq  = self.df.iloc[idx]['mutated']
        struct   = self.df.iloc[idx]['structure']
        seqs = [wild_seq, mut_seq, struct]
        seqs = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs]
        #mut_seq  = [" ".join(list(re.sub(r"[UZOB]", "X", mut_seq )))] #for sequence in mut_seq]
        #struct   = [" ".join(list(re.sub(r"[UZOB]", "X", struct  )))] #for sequence in struct]
        seqs = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in seqs]
        #mut_seq  = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in mut_seq]
        #struct   = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in struct]
        #seqs  = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest", return_tensors='pt')
        seqs  = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, max_length=self.max_length, padding="max_length", return_tensors='pt')
        pos = self.df.iloc[idx]['pos']
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        ddg = torch.unsqueeze(ddg, 0)
        return seqs, pos, ddg, wild_seq, mut_seq, struct

    def __len__(self):
        return len(self.df)




class ProteinDataLoader():
    def __init__(self, dataset, batch_size, num_workers, shuffle, pin_memory, sampler ):
        self.name = dataset.name
        self.df = dataset.df
        self.dataloader = DataLoader(dataset, 
                                     batch_size=batch_size, 
                                     num_workers=num_workers, 
                                     shuffle=shuffle, 
                                     pin_memory=pin_memory, 
                                     sampler=sampler)




 

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

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch):
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
        seqs, pos, labels, _,_,_ = batch
        seqs = seqs.to(self.local_rank)
        labels = labels.to(self.local_rank).reshape((-1,1))
        pos    =    pos.to(self.local_rank)
        with autocast(dtype=torch.bfloat16):
            logits = self.model(seqs, pos).to(self.local_rank)
            loss   = self.loss_fn(logits, labels)
        torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), max_norm = 0.2)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.scaler.step(self.optimizer)
        self.scheduler.step()
        self.scaler.update()
        self.t_labels.extend(labels.cpu().detach())
        self.t_preds.extend(logits.cpu().detach())

    def all_gather_lab_preds(self, preds, labels):
        #size is the world size
        size = get_world_size()
        preds = torch.tensor(preds).to(self.local_rank)
        labels = torch.tensor(labels).to(self.local_rank)
        prediction_list = [torch.zeros_like(preds).to(self.local_rank)  for _ in range(size)]
        labels_list     = [torch.zeros_like(labels).to(self.local_rank) for _ in range(size)]
        all_gather(prediction_list, preds)
        all_gather(labels_list, labels)
        new_preds  = torch.tensor([], dtype=torch.float32).to(self.local_rank)
        new_labels = torch.tensor([], dtype=torch.float32).to(self.local_rank)
        for t1 in prediction_list:
            new_preds = torch.cat((new_preds,t1), dim=0)
        for t2 in labels_list:
            new_labels = torch.cat((new_labels,t2),dim=0)
        return new_preds, new_labels 

    def train_epoch(self, epoch, train_proteindataloader):
        self.scaler = torch.cuda.amp.GradScaler()
        self.t_preds, self.t_labels = [], []
        self.model.train()
        len_dataloader = len(train_proteindataloader.dataloader)
        train_proteindataloader.dataloader.sampler.set_epoch(epoch)
        for idx, batch in enumerate(train_proteindataloader.dataloader):
            print_peak_memory("Max GPU memory", self.local_rank)
            print(f"{train_proteindataloader.name}\ton GPU {self.global_rank} epoch:{epoch+1}/{self.max_epochs}\tbatch_idx:{idx+1}/{len_dataloader}", flush=True)
            self.train_batch(batch)

        global_t_preds, global_t_labels = self.all_gather_lab_preds(self.t_preds, self.t_labels)
        g_t_labels = global_t_labels.to("cpu")
        g_t_preds  = global_t_preds.to("cpu")
        l_t_labels = torch.tensor(self.t_labels).to("cpu")
        l_t_preds  = torch.tensor(self.t_preds).to("cpu")
        return g_t_labels, g_t_preds, l_t_labels, l_t_preds

    def valid_batch(self, batch):
        seqs, pos, labels,_,_,_ = batch
        seqs   =   seqs.to(self.local_rank)
        labels = labels.to(self.local_rank)
        pos    =    pos.to(self.local_rank)
        logits = self.model(seqs, pos).to(self.local_rank)
        labels_cpu = labels.cpu().detach()
        logits_cpu = logits.cpu().detach()
        self.v_labels.extend(labels_cpu)
        self.v_preds.extend(logits_cpu)

    def valid_epoch(self, epoch, val_proteindataloader):
        self.model.eval()
        self.v_preds, self.v_labels = [], []
        len_dataloader = len(val_proteindataloader.dataloader)
        with torch.no_grad():
            for idx, batch in enumerate(val_proteindataloader.dataloader):
                print(f"{val_proteindataloader.name}\ton GPU {self.global_rank} epoch:{epoch+1}/{self.max_epochs}\tbatch_idx:{idx+1}/{len_dataloader}", flush=True)
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
                v_diffs.write(f"mut_seq,wild_seq,pos,ddg,pred\n")
            self.model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(val_dl.dataloader):
                    seqs, pos, labels,wild_seq, mut_seq, struct = batch
                    seqs   =   seqs.to(self.local_rank)
                    labels = labels.to(self.local_rank)
                    pos    =    pos.to(self.local_rank)
                    logits = self.model(seqs, pos).to(self.local_rank)
                    labels_cpu = labels.cpu().detach()
                    logits_cpu = logits.cpu().detach()
                    pos_cpu = pos.cpu().detach()
                    with open(self.result_dir + f"/{val_dl.name}_labels_preds.{self.global_rank}.diffs", "a") as v_diffs:
                       v_diffs.write(f"{mut_seq},{wild_seq},{pos_cpu},{labels_cpu},{logits_cpu}\n")

    def train(self, model, train_dl, val_dls):
        print(f"I am rank {self.local_rank}")
        self.model_name = model.name
        self.model = model.to(self.local_rank)
        self.model = DDP(self.model, device_ids=[self.local_rank])
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
            print(f"{self.train_dl.name}: on GPU {self.global_rank} epoch {epoch+1}/{self.max_epochs}\t"
                  f"rmse = {g_t_rmse} / {l_t_rmse}\t"
                  f"mae = {g_t_mae} / {l_t_mae}\t"
                  f"corr = {g_t_corr} / {l_t_corr}")
            if self.global_rank == 0:
                with open(self.train_logfile, "a") as t_log:
                    t_log.write(f"{epoch+1},{self.train_rmse[epoch]},{self.train_mae[epoch]},{self.train_corr[epoch]}\n")
            
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            
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
                      f"corr = {self.val_maes[val_no,epoch]} / {l_v_corr}")
                if self.global_rank == 0:
                    with open(self.result_dir + f"/{val_dl.name}_metrics.log", "a") as v_log:
                        v_log.write(f"{epoch+1},{self.val_rmses[val_no,epoch]},{self.val_maes[val_no,epoch]},{self.val_corrs[val_no,epoch]}\n")
        self.difference_labels_preds(cutoff=0.0)
        if self.global_rank == 0:
            self._save_snapshot(epoch)
        

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
        if self.global_rank == 0:
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
             
            print(self.val_maes)
            print(self.val_rmses)
            print(self.val_corrs)
            train_rmse_df["epoch"] = train_rmse_df.index
            train_mae_df["epoch"]  = train_mae_df.index
            train_corr_df["epoch"] = train_corr_df.index
            val_corr_df["epoch"]   = val_corr_df.index
            val_mae_df["epoch"]    = val_mae_df.index
            val_rmse_df["epoch"]   = val_rmse_df.index

            df = pd.concat([frame.set_index("epoch") for frame in [train_rmse_df, train_mae_df, train_corr_df, val_rmse_df, val_mae_df, val_corr_df]],
               axis=1, join="inner").reset_index()

            print(df, flush=True)

            df.to_csv( self.result_dir + "/epochs_statistics.csv")
            self.plot(df, train_rmse_name, val_rmse_names, ylabel="rsme", title="Model: {self.model_name}")
            self.plot(df, train_mae_name,  val_mae_names,  ylabel="mae",  title="Model: {self.model_name}")
            self.plot(df, train_corr_name, val_corr_names, ylabel="corr", title="Model: {self.model_name}")

    def free_memory(self):
        del self.model
        torch.cuda.empty_cache()

