#import argparse
#import math
#import matplotlib.pyplot as plt
#from matplotlib import cm
#import numpy as np
import os
#import pandas as pd
#import random
#import re
#import scipy
#from scipy import stats
#from scipy.stats import pearsonr
##from transformers import T5Tokenizer, T5EncoderModel
#import torch
#from torch import nn
#from torch.utils.data import Dataset, DataLoader
#import torch.nn.functional as F
#from torch.cuda.amp import autocast
#import yaml
#import sys
from models.models import *
from dataloader  import *
from utils  import *
from argparser import *
from tester import *


def main(output_dir, dataset_dir, model_name, max_length, max_tokens, snapshot_file):
    test_dir = dataset_dir + "/test"

    if model_name.rsplit("_")[0]=="ProstT5":
        test_dfs, test_names = from_cvs_files_in_dir_to_dfs_list(test_dir, datasets_dir="/translated_databases")
        test_dss = [ProstT5_Dataset(df=test_df,name=test_name,max_length=max_length) for test_df, test_name in zip(test_dfs, test_names)]
        collate_function = None
    if model_name.rsplit("_")[0]=="ESM2":
        test_dfs, test_names = from_cvs_files_in_dir_to_dfs_list(test_dir, datasets_dir="/databases")
        test_dss = [ESM2_Dataset(df=test_df,name=test_name,max_length=max_length) for test_df, test_name in zip(test_dfs, test_names)]
        collate_function = custom_collate
    if model_name.rsplit("_")[0]=="MSA":
        test_dfs, test_names = from_cvs_files_in_dir_to_dfs_list(test_dir, datasets_dir="/databases")
        test_dss = [MSA_Dataset(df=test_df,name=test_name, dataset_dir=test_dir, max_length=max_length,
                                max_tokens=max_tokens) for test_df, test_name in zip(test_dfs, test_names)] 
        collate_function = custom_collate

    test_dls = [ProteinDataLoader(test_ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False,sampler=None,custom_collate_fn=collate_function) for test_ds in test_dss]
    print(f"output_dir:\t{output_dir}\t{type(output_dir)}", flush=True)
    print(f"model_name:\t{model_name}\t{type(model_name)}", flush=True)
    print(f"snapshot_file:\t{snapshot_file}\t{type(snapshot_file)}", flush=True)
    print(f"test_dir:\t{test_dir}\t{type(test_dir)}", flush=True)
    print(f"max_length:\t{max_length}\t{type(max_length)}", flush=True)
    print(f"max_tokens:\t{max_tokens}\t{type(max_tokens)}", flush=True)
    tester     = Tester(output_dir=output_dir)
    test_model = models[model_name]()
    tester.test(test_model=test_model, test_dls=test_dls, snapshot_file=snapshot_file)

if __name__ == "__main__":
    args = argparser_tester()
    config_file = args.config_file
    if os.path.exists(config_file):
        config         = load_config(config_file)
        output_dir     = config["output_dir"]
        dataset_dir    = config["dataset_dir"]
        model_name     = config["model"]
        max_length     = config["max_length"]
        snapshot_file  = config["snapshot_file"]
        max_tokens     = config["MSA"]["max_tokens"]
    else:
        output_dir     = args.output_dir
        dataset_dir    = args.dataset_dir
        model_name     = args.model
        max_length     = args.max_length
        snapshot_file  = args.snapshot_file
        max_tokens     = args.max_tokens
    main(output_dir, dataset_dir, model_name, max_length, max_tokens, snapshot_file)


