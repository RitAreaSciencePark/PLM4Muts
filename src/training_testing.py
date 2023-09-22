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


def main(device_name, loss_fn_name, model_name, optimizer_name, train_dir, val_dir, lr, max_epochs, save_every):
    device = torch.device("cuda:0") if torch.cuda.is_available() and device_name == "cuda" else "cpu"
    curr_work_dir = os.getcwd()
    print(f"curr_work_dir={curr_work_dir}")
    loss_fn = losses[loss_fn_name]
    model = models[model_name]()
    optimizer = optimizers[optimizer_name](params=model.parameters(), lr=lr)
    
    train_dfs, _ = from_cvs_files_in_dir_to_dfs_list(curr_work_dir + "/" + train_dir)
    train_name = train_dir.rsplit('/', 1)[1] 
    train_df = pd.concat(train_dfs)
    train_ds = ProteinDataset(train_df, train_name)
    train_dl = ProteinDataLoader(train_ds, batch_size=1, num_workers = 0, shuffle = True, pin_memory=False, sampler=None)
    
    val_dfs, val_names = from_cvs_files_in_dir_to_dfs_list(curr_work_dir + "/" + val_dir)
    val_dss = [ProteinDataset(val_df, val_name) for val_df, val_name in zip(val_dfs, val_names) ] 
    val_dls = [ProteinDataLoader(val_ds, batch_size=1, num_workers = 0, shuffle = False, pin_memory=False, sampler=None) for val_ds in val_dss]
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl.dataloader), epochs=max_epochs)
    
    trainer = Trainer(max_epochs=max_epochs,loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
                  gpu_id=0, save_every=save_every, current_dir=current_dir)

    trainer.train(model=model,  train_dl=train_dl, val_dls=val_dls)
    trainer.describe()

if __name__ == "__main__":
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

    main(device_name, loss_fn_name, model_name, optimizer_name, train_dir, val_dir, lr, max_epochs, save_every)
