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
import yaml
import sys
from models.models import *
from utils.data_train_val  import *
from utils.argparser import *
from torch.distributed import init_process_group, destroy_process_group, barrier, get_rank

# Global dictionaries for Models, Losses and Optimizers
models = {"Milano":          ProstT5_Milano,
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


def main(loss_fn_name, model_name, optimizer_name, train_dir, val_dir, lr, max_epochs, save_every, output_dir):
    ddp_setup()
    loss_fn   = losses[loss_fn_name]
    model     = models[model_name]()
    optimizer = optimizers[optimizer_name](params=model.parameters(), lr=lr)
    train_dfs, _ = from_cvs_files_in_dir_to_dfs_list(train_dir)
    train_df     = pd.concat(train_dfs)
    train_name   = train_dir.rsplit('/', 1)[1] 
    train_ds     = ProteinDataset(train_df, train_name)
    train_dl     = ProteinDataLoader(train_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=True, sampler=DistributedSampler(train_ds))
    val_dfs,val_names = from_cvs_files_in_dir_to_dfs_list(val_dir)
    val_dss           = [ProteinDataset(val_df, val_name) for val_df, val_name in zip(val_dfs, val_names)] 
    val_dls           = [ProteinDataLoader(val_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, sampler=DistributedSampler(val_ds)) for val_ds in val_dss]
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl.dataloader), epochs=max_epochs)
    trainer   = Trainer(max_epochs=max_epochs,loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, save_every=save_every, output_dir=output_dir)
    if int(os.environ["RANK"]) == 0:
        print(f"output_dir:\t{output_dir}\t{type(output_dir)}", flush=True)
        print(f"loss_fn_name:\t{loss_fn_name}\t{type(loss_fn_name)}", flush=True)
        print(f"learning rate:\t{lr}\t{type(lr)}", flush=True)
        print(f"max_epochs:\t{max_epochs}\t{type(max_epochs)}", flush=True)
        print(f"model_name:\t{model_name}\t{type(model_name)}", flush=True)
        print(f"optimizer_name:\t{optimizer_name}\t{type(optimizer_name)}", flush=True)
        print(f"train_dir:\t{train_dir}\t{type(train_dir)}", flush=True)
        print(f"val_dir:\t{val_dir}\t{type(val_dir)}", flush=True)

    trainer.train(model=model,  train_dl=train_dl, val_dls=val_dls)
    trainer.describe()
    destroy_process_group()


if __name__ == "__main__":
    # Define the args from argparser
    args = argparser()
    config_file = args.config_file
    # load config file
    if os.path.exists(config_file):
        config = load_config(config_file)
        output_dir     = config["output_dir"]
        loss_fn_name   = config["loss_fn"]
        lr             = config["learning_rate"]
        max_epochs     = config["max_epochs"]
        model_name     = config["model"]
        optimizer_name = config["optimizer"]
        train_dir      = config["train_dir"]
        val_dir        = config["val_dir"]
        save_every     = config["save_every"] 
    else:
        output_dir     = args.output_dir
        lr             = args.lr
        max_epochs     = args.max_epochs
        loss_fn_name   = args.loss_fn
        model_name     = args.model
        optimizer_name = args.optimizer
        train_dir      = args.train_dir
        val_dir        = args.val_dir
        save_every     = args.save_every

    main(loss_fn_name, model_name, optimizer_name, train_dir, val_dir, lr, max_epochs, save_every, output_dir)

