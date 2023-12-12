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
from dataloader  import *
from utils  import *
from trainer  import *
from argparser import *
import torch.distributed  as dist
from torch.utils.data.distributed import DistributedSampler

# Global dictionaries for Models, Losses and Optimizers
models = {"ProstT5_Torino":          ProstT5_Torino,
          "ProstT5_Roma":            ProstT5_Roma,
          "MSA_Torino":              MSA_Torino,
          "MSA_Trieste":             MSA_Trieste,
          "ESM_Torino":              ESM_Torino,
          "ProstT5_Milano":          ProstT5_Milano
        }

losses = {"L1":  torch.nn.functional.l1_loss,
          "MSE": torch.nn.functional.mse_loss,
         }

optimizers = {"Adam":  torch.optim.Adam,
              "AdamW": torch.optim.AdamW, 
             }


def main(output_dir,dataset_dir, 
         model_name, max_epochs, loss_fn_name, device, max_length,    
         lr, optimizer_name, weight_decay,
         max_tokens):

    ddp_setup()
    # fix the seed for reproducibility
    seed = 10 * (11 + 12 * dist.get_rank())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    loss_fn   = losses[loss_fn_name]
    model     = models[model_name]()
    if optimizer_name=="AdamW":
        optimizer = optimizers[optimizer_name](params=model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name=="Adam":
        optimizer = optimizers[optimizer_name](params=model.parameters(), lr=lr)
    train_dir = dataset_dir + "/train"
    test_dir  = dataset_dir + "/test"
    train_dfs, _ = from_cvs_files_in_dir_to_dfs_list(train_dir)
    train_df     = pd.concat(train_dfs)
    train_name   = dataset_dir.rsplit('/', 1)[1] + "_training"
    val_dfs, val_names = from_cvs_files_in_dir_to_dfs_list(test_dir)
    
    if model_name.rsplit("_")[0]=="ProstT5":
        train_ds =  ProteinDataset(df=train_df, name=train_name, max_length=max_length)
        val_dss  = [ProteinDataset(df=val_df,   name=val_name,   max_length=max_length) for val_df, val_name in zip(val_dfs, val_names)]
        test_dss = [ProteinDataset(df=val_df,   name=val_name,   max_length=max_length) for val_df, val_name in zip(val_dfs, val_names)]
        collate_function = None

    if model_name.rsplit("_")[0]=="ESM": 
        train_ds =  ESM_Dataset(df=train_df, name=train_name, max_length=max_length)
        val_dss  = [ESM_Dataset(df=val_df,   name=val_name,   max_length=max_length) for val_df, val_name in zip(val_dfs, val_names)]
        test_dss = [ESM_Dataset(df=val_df,   name=val_name,   max_length=max_length) for val_df, val_name in zip(val_dfs, val_names)]
        collate_function = custom_collate

    if model_name.rsplit("_")[0]=="MSA":
        train_ds =  MSA_Dataset(df=train_df, name=train_name, dataset_dir=train_dir, max_length=max_length, 
                                max_tokens=max_tokens)

        val_dss  = [MSA_Dataset(df=val_df,   name=val_name,   dataset_dir=test_dir, max_length=max_length, 
                                max_tokens=max_tokens) for val_df, val_name in zip(val_dfs, val_names)] 
        
        test_dss = [MSA_Dataset(df=val_df,   name=val_name,   dataset_dir=test_dir, max_length=max_length, 
                                max_tokens=max_tokens) for val_df, val_name in zip(val_dfs, val_names)] 
        
        collate_function = custom_collate

    train_dl = ProteinDataLoader(train_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False,
                                 sampler=DistributedSampler(train_ds),  custom_collate_fn=collate_function)
    val_dls  = [ProteinDataLoader(val_ds,  batch_size=1, num_workers=0, shuffle=False, pin_memory=False, 
                                  sampler=DistributedSampler(val_ds),   custom_collate_fn=collate_function) for val_ds in val_dss]
    test_dls = [ProteinDataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False,
                                  sampler=None, custom_collate_fn=collate_function) for test_ds in test_dss]
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl.dataloader), epochs=max_epochs)
    trainer   = Trainer(max_epochs=max_epochs, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, output_dir=output_dir)
    if int(os.environ["RANK"]) == 0:
        print(f"output_dir:\t{output_dir}\t{type(output_dir)}", flush=True)
        print(f"loss_fn_name:\t{loss_fn_name}\t{type(loss_fn_name)}", flush=True)
        print(f"learning rate:\t{lr}\t{type(lr)}", flush=True)
        print(f"max_epochs:\t{max_epochs}\t{type(max_epochs)}", flush=True)
        print(f"model_name:\t{model_name}\t{type(model_name)}", flush=True)
        print(f"optimizer_name:\t{optimizer_name}\t{type(optimizer_name)}", flush=True)
        print(f"train_dir:\t{train_dir}\t{type(train_dir)}", flush=True)
        print(f"test_dir:\t{test_dir}\t{type(test_dir)}", flush=True)
        print(f"max_length:\t{max_length}\t{type(max_length)}", flush=True)
        print(f"max_tokens:\t{max_tokens}\t{type(max_tokens)}", flush=True)
        print(f"weight_decay:\t{weight_decay}\t{type(weight_decay)}", flush=True)
    trainer.train(model=model, train_dl=train_dl, val_dls=val_dls)
    trainer.describe()
    dist.barrier()
    trainer.free_memory(model)
    dist.destroy_process_group()

if __name__ == "__main__":
    # Define the args from argparser
    args = argparser_trainer()
    config_file = args.config_file
    # load config file
    if os.path.exists(config_file):
        config = load_config(config_file)
        output_dir     = config["output_dir"]
        dataset_dir    = config["dataset_dir"]
        model_name     = config["model"]
        max_epochs     = config["max_epochs"]
        loss_fn_name   = config["loss_fn"]
        device         = config["device"]
        max_length     = config["max_length"]
        lr             = config["learning_rate"]
        optimizer_name = config["optimizer"]["name"]
        weight_decay   = config["optimizer"]["weight_decay"]
        max_tokens     = config["MSA"]["max_tokens"]
    else:
        output_dir     = args.output_dir
        dataset_dir    = args.dataset_dir
        model_name     = args.model
        max_epochs     = args.max_epochs
        loss_fn_name   = args.loss_fn
        device         = args.device
        max_length     = args.max_length
        lr             = args.learning_rate
        optimizer_name = args.optimizer
        weight_decay   = args.weight_decay
        max_tokens     = args.max_tokens
    main(output_dir,dataset_dir, 
         model_name, max_epochs, loss_fn_name, device, max_length, 
         lr, optimizer_name, weight_decay, 
         max_tokens)

