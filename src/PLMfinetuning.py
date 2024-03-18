#import argparse
#import math
#import matplotlib.pyplot as plt
#from matplotlib import cm
#import numpy as np
import os
from pathlib import Path
#import pandas as pd
import random
#import re
#import scipy
#from scipy import stats
#from scipy.stats import pearsonr
#from transformers import T5Tokenizer, T5EncoderModel
import torch
#from torch import nn
from torch.utils.data import Dataset, DataLoader
#import torch.nn.functional as F
#from torch.cuda.amp import autocast
#import yaml
#import sys
from models.models import *
from dataloader  import *
from utils  import *
from trainer  import *
from argparser import *
import torch.distributed  as dist
from torch.utils.data.distributed import DistributedSampler


def main(output_dir,dataset_dir, 
         model_name, max_epochs, loss_fn_name, max_length, 
         lr, seeds, optimizer_name, weight_decay, momentum,
         max_tokens):

    ddp_setup()
    #seeds = (10,   11,   12)
    #seeds = (100,  110,  120)
    #seeds = (1000, 1100, 1200)
    # fix the seed for reproducibility
    seed = int(seeds[0]) * (int(seeds[1]) + int(seeds[2]) * dist.get_rank())
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"seed={seed} on GPU {dist.get_rank()}")

    loss_fn   = losses[loss_fn_name]
    model     = models[model_name]()
    if optimizer_name=="SGD":
        optimizer = optimizers[optimizer_name](params=model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    if optimizer_name=="AdamW":
        optimizer = optimizers[optimizer_name](params=model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name=="Adam":
        optimizer = optimizers[optimizer_name](params=model.parameters(), lr=lr)
    train_dir = dataset_dir + "/train"
    val_dir   = dataset_dir + "/validation"
    test_dir  = dataset_dir + "/test"
    
    if model_name.rsplit("_")[0]=="ProstT5":
        train_dfs, _ = from_cvs_files_in_dir_to_dfs_list(train_dir, datasets_dir="/translated_databases")
        train_df     = pd.concat(train_dfs)
        train_name   = dataset_dir.rsplit('/', 1)[1] + "_training"
        val_dfs,   val_names = from_cvs_files_in_dir_to_dfs_list(val_dir, datasets_dir="/translated_databases")
        test_dfs, test_names = from_cvs_files_in_dir_to_dfs_list(test_dir, datasets_dir="/translated_databases")
        train_ds=ProstT5_Dataset(df=train_df,name=train_name,max_length=max_length)
        val_dss=[ProstT5_Dataset(df=val_df,name=val_name,max_length=max_length) for val_df,val_name in zip(val_dfs,val_names)]
        test_dss=[ProstT5_Dataset(df=test_df,name=test_name,max_length=max_length) for test_df,test_name in zip(test_dfs,test_names)]
        collate_function = None

    if model_name.rsplit("_")[0]=="ESM2": 
        train_dfs, _ = from_cvs_files_in_dir_to_dfs_list(train_dir, datasets_dir="/databases")
        train_df     = pd.concat(train_dfs)
        train_name   = dataset_dir.rsplit('/', 1)[1] + "_training"
        val_dfs,   val_names = from_cvs_files_in_dir_to_dfs_list(val_dir, datasets_dir="/databases")
        test_dfs, test_names = from_cvs_files_in_dir_to_dfs_list(test_dir, datasets_dir="/databases")
        train_ds=ESM2_Dataset(df=train_df,name=train_name,max_length=max_length)
        val_dss=[ESM2_Dataset(df=val_df,name=val_name,max_length=max_length) for val_df,val_name in zip(val_dfs,val_names)]
        test_dss=[ESM2_Dataset(df=test_df,name=test_name, max_length=max_length) for test_df,test_name in zip(test_dfs,test_names)]
        collate_function = custom_collate

    if model_name.rsplit("_")[0]=="MSA":
        train_dfs, _ = from_cvs_files_in_dir_to_dfs_list(train_dir, datasets_dir="/databases")
        train_df     = pd.concat(train_dfs)
        train_name   = dataset_dir.rsplit('/', 1)[1] + "_training"
        val_dfs,   val_names = from_cvs_files_in_dir_to_dfs_list(val_dir, datasets_dir="/databases")
        test_dfs, test_names = from_cvs_files_in_dir_to_dfs_list(test_dir, datasets_dir="/databases")
        train_ds =  MSA_Dataset(df=train_df, name=train_name, dataset_dir=train_dir, max_length=max_length, 
                                max_tokens=max_tokens)

        val_dss  = [MSA_Dataset(df=val_df,   name=val_name,   dataset_dir=val_dir, max_length=max_length, 
                                max_tokens=max_tokens) for val_df, val_name in zip(val_dfs, val_names)] 
        
        test_dss = [MSA_Dataset(df=test_df,   name=test_name,   dataset_dir=test_dir, max_length=max_length, 
                                max_tokens=max_tokens) for test_df, test_name in zip(test_dfs, test_names)] 
        
        collate_function = custom_collate

    train_dl = ProteinDataLoader(train_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False,
                                 sampler=DistributedSampler(train_ds),  custom_collate_fn=collate_function)
    val_dls  = [ProteinDataLoader(val_ds,  batch_size=1, num_workers=0, shuffle=False, pin_memory=False, 
                                  sampler=DistributedSampler(val_ds),   custom_collate_fn=collate_function) for val_ds in val_dss]
    test_dls = [ProteinDataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False,
                                  sampler=DistributedSampler(test_ds), custom_collate_fn=collate_function) for test_ds in test_dss]
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dl.dataloader), epochs=max_epochs)
    trainer   = Trainer(max_epochs=max_epochs, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, output_dir=output_dir, seeds=seeds)
    
    dist.barrier()
    if int(os.environ["RANK"]) == 0:
        print(f"[Info] output_dir:\t{output_dir}\t{type(output_dir)}", flush=True)
        print(f"[Info] loss_fn_name:\t{loss_fn_name}\t{type(loss_fn_name)}", flush=True)
        print(f"[Info] learning rate:\t{lr}\t{type(lr)}", flush=True)
        print(f"[Info] seeds:\t{seeds}\t{type(seeds)}", flush=True)
        print(f"[Info] max_epochs:\t{max_epochs}\t{type(max_epochs)}", flush=True)
        print(f"[Info] model_name:\t{model_name}\t{type(model_name)}", flush=True)
        print(f"[Info] optimizer_name:\t{optimizer_name}\t{type(optimizer_name)}", flush=True)
        print(f"[Info] train_dir:\t{train_dir}\t{type(train_dir)}", flush=True)
        print(f"[Info] val_dir:\t{val_dir}\t{type(val_dir)}", flush=True)
        print(f"[Info] test_dir:\t{test_dir}\t{type(test_dir)}", flush=True)
        print(f"[Info] max_length:\t{max_length}\t{type(max_length)}", flush=True)
        print(f"[Info] max_tokens:\t{max_tokens}\t{type(max_tokens)}", flush=True)
        print(f"[Info] weight_decay:\t{weight_decay}\t{type(weight_decay)}", flush=True)
        print(f"[Info] momentum:\t{momentum}\t{type(momentum)}", flush=True)
        ft_start_time = get_date_of_run()
        ft_start_time_str = ft_start_time.strftime("%Y-%m-%d-%I:%M:%S_%p")
        print(f"[Info] Model Finetuning started at: {ft_start_time_str}")
    dist.barrier()
    trainer.train(model=model, train_dl=train_dl, val_dls=val_dls, test_dls=test_dls)
    dist.barrier()
    if int(os.environ["RANK"]) == 0:
        ft_end_time = get_date_of_run()
        ft_end_time_str = ft_end_time.strftime("%Y-%m-%d-%I:%M:%S_%p")
        print(f"[Info] Model Finetuning completed at: {ft_start_time_str}")
        ft_duration = ft_end_time - ft_start_time
        print(f"[Info] Total time for Fine-tuning: {ft_duration}")

    dist.barrier()
    trainer.describe()
    dist.barrier()
    trainer.free_memory(model)
    dist.destroy_process_group()

if __name__ == "__main__":
    args = argparser_trainer()
    target = args.config_file
    target_path = Path(target)
    if not os.path.exists(target_path):
        print(f"The path {target_path} doesn't exist")
        raise SystemExit(1)

    if os.path.isfile(target_path):
        print(f"Opening the configuration file {target_path}")
        config = load_config(target_path)
        try:
            dataset_dir = config["dataset_dir"]
            dataset_path = Path(dataset_dir)
            if not os.path.exists(dataset_path):
                print(f"The dataset path {dataset_path} doesn't exist")
                raise SystemExit(1)
            if not os.path.isdir(dataset_path):
                print(f"The dataset path {dataset_path} is not a directory")
                raise SystemExit(1)
        except:
            print(f"The dataset directory doesn't exist")
            raise SystemExit(1)
        try:
            output_dir = config["output_dir"]
        except:
            output_dir = os.getcwd()
            print(f"Setting the default output directory: {output_dir}")
        try:    
            model_name = config["model"]
        except:
            model_name = "MSA_Finetuning"
            print(f"Setting the default model: {model_name}")
        try:
            max_epochs = config["max_epochs"]
        except:
            max_epochs = 3
            print(f"Setting the default max number of training epochs: {max_epochs}")
        try:
            loss_fn_name = config["loss_fn"]
        except:
            loss_fn_name = "L1"
            print(f"Setting the default {loss_fn_name} loss")
        try:
            max_length = config["max_length"]
        except:
            max_length = 1024
            print(f"Setting the default max length of the aminoacid sequence: {max_length}")
        try:
            lr = config["learning_rate"]
        except:
            lr = 5.0e-6
            print(f"Setting the default learning rate: {lr}")
        try:
            seeds = config["seeds"]
        except:
            seeds = [10, 11, 12]
            print(f"Setting the default seeds: {seeds}")
        try:
            optimizer_name = config["optimizer"]["name"]
        except:
            optimizer_name = "AdamW"
            print(f"Setting the default optimizer: {optimizer_name}")
        try:
            weight_decay = config["optimizer"]["weight_decay"]
        except:
            weight_decay = 0.01
            if optimizer_name == "AdamW" or optimizer_name == "SGD":
                print(f"Setting the default weight decay: {weight_decay}")
        try:
            momentum = config["optimizer"]["momentum"]
        except:
            momentum = 0.
            if optimizer_name == "SGD":
                print(f"Setting the default momentum: {momentum}")
        try:
            max_tokens = config["MSA"]["max_tokens"]
        except:
            max_tokens = 16000
            if model_name == "MSA_Finetuning" or model_name == "MSA_Baseline":
                print(f"Setting the default max number of tokens for MSA: {max_tokens}")

    else:
        print(f"The path {target_path} is not valid")
        raise SystemExit(1)

    main(output_dir,dataset_dir, 
         model_name, max_epochs, loss_fn_name, max_length, 
         lr, seeds, optimizer_name, weight_decay, momentum,
         max_tokens)

