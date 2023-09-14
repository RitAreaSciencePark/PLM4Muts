import argparse
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import pandas as pd
import random
import re
import scipy
from scipy import stats
from scipy.stats import pearsonr
from transformers import T5Tokenizer, T5EncoderModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast
import warnings
import yaml

from models.models import *
from utils.data_train_val  import *
from utils.argparser import *

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

HIDDEN_UNITS_POS_CONTACT = 5

# CUDA specifications

print("\ntorch.cuda.is_available() =", torch.cuda.is_available(), "\ttorch version =", torch.version.cuda)

# Set Random Seeds

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Global dictionaries for Models, Losses and Optimizers

models = {"Milano":          ProstT5_Milano,
          "MilanoMean":      ProstT5_MilanoMean,
          "Roma":            ProstT5_Roma,
          "RomaMean":        ProstT5_RomaMean,
          "Trieste":         ProstT5_Trieste,
          "TriesteMean":     ProstT5_TriesteMean,
          "Conconello":      ProstT5_Conconello,
          "ConconelloMean":  ProstT5_ConconelloMean,
          "Basovizza":       ProstT5_Basovizza,
          "BasovizzaMean":   ProstT5_BasovizzaMean,
          "Padriciano":      ProstT5_Padriciano,
          "PadricianoMean":  ProstT5_PadricianoMean,
          "mutLin2":         ProstT5_mutLin2,
          "mutLin2Mean":     ProstT5_mutLin2Mean,
          "mutLin4":         ProstT5_mutLin4,
          "mutLin4Mean":     ProstT5_mutLin4Mean,
        }

losses = {"L1":  torch.nn.functional.l1_loss,
          "MSE": torch.nn.functional.mse_loss,
          }

optimizers = {"Adam":  torch.optim.Adam,
              "AdamW": torch.optim.AdamW, 
              }

# Define the args from argparser

args = argparser()
current_dir = args.current_dir

# load config file
config_file = os.path.join(current_dir, "config.yaml") 
if os.path.exists(config_file):
    config = load_config(config_file)
    device_name    = config["device"]
    loss_fn_name   = config["loss_fn"]
    lr             = config["learning_rate"]
    max_epochs     = config["max_epochs"]
    model_name     = config["model"]
    optimizer_name = config["optimizer"]
    train_dir      = config["train_dir"]
    val_dir        = config["val_dir"]
    save_every     = config["save_every"] 
else:
    lr = args.lr
    max_epochs     = args.max_epochs
    loss_fn_name   = args.loss_fn
    model_name     = args.model
    optimizer_name = args.optimizer
    train_dir      = args.train_dir
    val_dir        = args.val_dir
    device_name    = args.device
    save_every     = args.save_every

print(f"device_name:\t{device_name}\t{type(device_name)}", flush=True)
print(f"loss_fn_name:\t{loss_fn_name}\t{type(loss_fn_name)}", flush=True)
print(f"learning rate:\t{lr}\t{type(lr)}", flush=True)
print(f"max_epochs:\t{max_epochs}\t{type(max_epochs)}", flush=True)
print(f"model_name:\t{model_name}\t{type(model_name)}", flush=True)
print(f"optimizer_name:\t{optimizer_name}\t{type(optimizer_name)}", flush=True)
print(f"train_dir:\t{train_dir}\t{type(train_dir)}", flush=True)
print(f"val_dir:\t{val_dir}\t{type(val_dir)}", flush=True)

# Main

device = torch.device("cuda") if torch.cuda.is_available() and device_name == "cuda" else "cpu"

result_dir = current_dir + "/results"
if not(os.path.exists(result_dir) and os.path.isdir(result_dir)):
    os.makedirs(result_dir)

curr_work_dir = os.getcwd()

print(f"curr_work_dir={curr_work_dir}")

loss_fn = losses[loss_fn_name]
model = models[model_name]()
model.to(device)
optimizer = optimizers[optimizer_name](params=model.parameters(), lr=lr)

train_dfs, _       = from_cvs_files_in_dir_to_dfs_list(curr_work_dir + "/" + train_dir)
train_name =  train_dir.rsplit('/', 1)[1] 
val_dfs, val_names = from_cvs_files_in_dir_to_dfs_list(curr_work_dir + "/" + val_dir)

print(train_name, val_names)
train_df = pd.concat(train_dfs)

train_ds    = ProteinDataset(train_df, train_name)
train_dl    = ProteinDataLoader(train_ds, batch_size=1, num_workers = 0, shuffle = True, pin_memory=False, sampler=None)
#train_rmse  = np.zeros(max_epochs) 
#train_mae   = np.zeros(max_epochs) 
#train_corr  = np.zeros(max_epochs)
scheduler   = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl.dataloader), epochs=max_epochs)
    
val_dss   = [ProteinDataset(val_df, val_name) for val_df, val_name in zip(val_dfs, val_names) ] 
val_dls   = [ProteinDataLoader(val_ds, batch_size=1, num_workers = 0, shuffle = False, pin_memory=False, sampler=None) for val_ds in val_dss]

trainer = Trainer(max_epochs=max_epochs,loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                  device=device, gpu_id=0, save_every=save_every, current_dir=current_dir)

trainer.train(model=model,  train_dl=train_dl, val_dls=val_dls)
trainer.describe()

#val_rmses = [np.zeros(max_epochs)] * len(val_dls)
#val_maes  = [np.zeros(max_epochs)] * len(val_dls)
#val_corrs = [np.zeros(max_epochs)] * len(val_dls)

#for epoch in range(max_epochs):
#    t_labels, t_preds = train_epoch(model, train_dl, device, optimizer, scheduler, epoch)
#    t_mse  = np.mean(      (np.array(t_labels) - np.array(t_preds))**2)  
#    t_mae  = np.mean(np.abs(np.array(t_labels) - np.array(t_preds))   )
#    t_rmse = np.sqrt(t_mse)
#    t_corr, _ = pearsonr(t_labels, t_preds)
#    train_mae[epoch]  = t_mae
#    train_corr[epoch] = t_corr
#    train_rmse[epoch] = t_rmse
#    print("***********************************************************************", flush=True) 
#    print(f"Training Dataset - {model_name}:\tlen={len(train_dl)}", flush=True) 
#    print(f"Training RMSE - {model_name}            for epoch {epoch+1}/{max_epochs}:\t{train_rmse[epoch]}", flush=True) 
#    print(f"Training MAE - {model_name}             for epoch {epoch+1}/{max_epochs}:\t{train_mae[epoch]}", flush=True) 
#    print(f"Training Correlation - {model_name}     for epoch {epoch+1}/{max_epochs}:\t{train_corr[epoch]}", flush=True) 
#    print("***********************************************************************", flush=True) 
#    for val_no, val_dl in enumerate(val_dls):
#        v_labels, v_preds = valid_epoch(model, val_dl, device)
#        v_mse  = np.mean(      (np.array(v_labels) - np.array(v_preds))**2) 
#        v_mae  = np.mean(np.abs(np.array(v_labels) - np.array(v_preds))   )
#        v_rmse = np.sqrt(v_mse)
#        v_corr, _ = pearsonr(v_labels, v_preds)
#        val_maes[val_no][epoch]  = v_mae
#        val_corrs[val_no][epoch] = v_corr
#        val_rmses[val_no][epoch] = v_rmse            
#        print("***********************************************************************", flush=True) 
#        print(f"Validation Dataset - {model_name}:\t{val_names[val_no]} len={len(val_dl)}", flush=True) 
#        print(f"Validation RMSE - {model_name}            for epoch {epoch+1}/{max_epochs}:\t{val_rmses[val_no][epoch]}", flush=True) 
#        print(f"Validation MAE - {model_name}             for epoch {epoch+1}/{max_epochs}:\t{val_maes[val_no][epoch]}", flush=True) 
#        print(f"Validation Correlation - {model_name}     for epoch {epoch+1}/{max_epochs}:\t{val_corrs[val_no][epoch]}", flush=True) 
#        print("***********************************************************************", flush=True) 
           
#        if epoch==max_epochs - 1:
#            for idx,(lab, pred) in enumerate(zip(v_labels,v_preds)):
#                if np.abs(lab-pred) > 0.0:
#                    wild_seq = val_dfs[val_no].iloc[idx]['wild_type']
#                    mut_seq  = val_dfs[val_no].iloc[idx]['mutated']
#                    pos = val_dfs[val_no].iloc[idx]['pos']
#                    ddg = val_dfs[val_no].iloc[idx]['ddg']
#                    print(f"\n{val_names[val_no]}:\nwild_seq={wild_seq}\nmuta_seq={mut_seq}\npos={pos}\nlabels={lab}\tpredictions={pred}\n", 
#                              flush=True)
         
#model.to('cpu') 
        
#    torch.save(model.state_dict(), 'weights/' + model_name)
    
#del model
#torch.cuda.empty_cache()

#print("Summary Direct Training", flush=True) 

#val_rmse_names  = [ s + "_rmse" for s in val_names]
#val_mae_names   = [ s + "_mae"  for s in val_names]
#val_corr_names  = [ s + "_corr" for s in val_names]
#train_rmse_dict = {'train_rmse': train_rmse}
#train_mae_dict  = {'train_mae':  train_mae}
#train_corr_dict = {'train_corr': train_corr}

#val_rmse_df   = pd.DataFrame.from_dict(dict(zip(val_rmse_names,  val_rmses)))
#val_mae_df    = pd.DataFrame.from_dict(dict(zip(val_mae_names,   val_maes)))
#val_corr_df   = pd.DataFrame.from_dict(dict(zip(val_corr_names,  val_corrs)))
#train_rmse_df = pd.DataFrame.from_dict(train_rmse_dict)
#train_mae_df  = pd.DataFrame.from_dict(train_mae_dict)
#train_corr_df = pd.DataFrame.from_dict(train_corr_dict)

#train_rmse_df["epoch"] = train_rmse_df.index
#train_mae_df["epoch"]  = train_mae_df.index
#train_corr_df["epoch"] = train_corr_df.index
#val_corr_df["epoch"]   = val_corr_df.index
#val_mae_df["epoch"]    = val_mae_df.index
#val_rmse_df["epoch"]   = val_rmse_df.index

#df = pd.concat([frame.set_index("epoch") for frame in [train_rmse_df, train_mae_df, train_corr_df, val_rmse_df, val_mae_df, val_corr_df]],
#               axis=1, join="inner").reset_index()

#print(df, flush=True) 

#if not(os.path.exists(result_dir) and os.path.isdir(result_dir)):
#    os.makedirs(result_dir)

#df.to_csv( result_dir + "/epochs_statistics.csv")

#colors = cm.rainbow(torch.linspace(0, 1, len(val_rmse_names)))
#plt.figure(figsize=(8,6))
#plt.plot(df["epoch"], df["train_rmse"],      label='train_rmse',  color="black",   linestyle="-.")
#for i, val_rmse_name in enumerate(val_rmse_names): 
#    plt.plot(df["epoch"], df[val_rmse_name], label=val_rmse_name, color=colors[i], linestyle="-")
#plt.xlabel("epoch")
#plt.ylabel("RMSE")
#plt.title(f"Model: {model_name}")
#plt.legend()
#plt.savefig( result_dir + "/epochs_rmse.png")
#plt.clf()

#plt.figure(figsize=(8,6))
#plt.plot(df["epoch"], df["train_mae"],      label='train_mae',  color="black",   linestyle="-.")
#for i, val_mae_name in enumerate(val_mae_names): 
#    plt.plot(df["epoch"], df[val_mae_name], label=val_mae_name, color=colors[i], linestyle="-")
#plt.xlabel("epoch")
#plt.ylabel("MAE")
#plt.title(f"Model: {model_name}")
#plt.legend()
#plt.savefig( result_dir + "/epochs_mae.png")
#plt.clf()

#plt.figure(figsize=(8,6))
#plt.plot(df["epoch"], df["train_corr"],      label='train_corr',  color="black",   linestyle="-.")
#for i, val_corr_name in enumerate(val_corr_names): 
#    plt.plot(df["epoch"], df[val_corr_name], label=val_corr_name, color=colors[i], linestyle="-")
#plt.xlabel("epoch")
#plt.ylabel("Pearson correlation coefficient")
#plt.title(f"Model: {model_name}")
#plt.legend()
#plt.savefig(result_dir + "/epochs_pearsonr.png")
#plt.clf()

