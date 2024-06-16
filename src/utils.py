import pandas as pd
import os
import torch
from models.models import *
import datetime

# Global dictionaries for Models, Losses and Optimizers
models = {
          "ESM2_Finetuning":            ESM2_Finetuning,
          "ESM2_Finetuning_OnlyMean":   ESM2_Finetuning_OnlyMean,
          "ESM2_Finetuning_OnlyPos":    ESM2_Finetuning_OnlyPos,
          "ESM2_Finetuning_Logits":     ESM2_Finetuning_Logits,
          "ESM2_Baseline":              ESM2_Baseline,
          "MSA_Finetuning":             MSA_Finetuning,
          "MSA_Finetuning_OnlyMean":    MSA_Finetuning_OnlyMean,
          "MSA_Finetuning_OnlyPos":     MSA_Finetuning_OnlyPos,
          "MSA_Finetuning_Logits":      MSA_Finetuning_Logits,
          "MSA_Baseline":               MSA_Baseline,
          "ProstT5_Finetuning":         ProstT5_Finetuning,
          "ProstT5_Baseline":           ProstT5_Baseline,
        }

losses = {"L1":  torch.nn.functional.l1_loss,
          "MSE": torch.nn.functional.mse_loss,
         }

optimizers = {"Adam":  torch.optim.Adam,
              "AdamW": torch.optim.AdamW,
              "SGD":   torch.optim.SGD,
             }

def get_date_of_run():
    date_of_run = datetime.datetime.now()#.strftime("%Y-%m-%d-%I:%M:%S_%p")
    return date_of_run

def from_cvs_files_in_dir_to_dfs_list(main_dir, databases_dir="/databases"):
    dir_path = main_dir + databases_dir
    datasets = os.listdir(dir_path)
    datasets_names = [ s.rsplit('.', 1)[0]  for s in datasets ]
    dfs = [None] * len(datasets)
    for i,d in enumerate(datasets):
        d_path = os.path.join(dir_path, d)
        dfs[i] = pd.read_csv(d_path, sep=',')
    return dfs, datasets_names

