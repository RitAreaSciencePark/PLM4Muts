import argparse
import math
import matplotlib.pyplot as plt
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

from models.models import *
from utils.data_train_val  import *
from utils.argparser import *

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

HIDDEN_UNITS_POS_CONTACT = 5

print("\ntorch.cuda.is_available() =", torch.cuda.is_available(), "\ttorch version =", torch.version.cuda)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)




args = argparser()
print("\n", args)

# Main

lr = args.lr
EPOCHS = args.epochs
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

allmodels = ['ProstT5_Milano', 
             'ProstT5_Roma', 
             'ProstT5_Trieste', 
             'ProstT5_Conconello', 
             'ProstT5_Basovizza',
             'ProstT5_Padriciano',
             'ProstT5_mutLin2', 
             'ProstT5_mutLin4',
             ]

optimizer_name="Adam" # "AdamW"

models = [allmodels[5]]

training_name="fixed_training"
training_name="cut_training"
CurrWorDir = os.getcwd()
print()

#test_path  = CurrWorDir + "/S669_subsets/data/"
#test_files = os.listdir(test_path)

full_df = pd.read_csv('../datasets/' + training_name +'_direct.csv',sep=',')
test_datasets = ['datasets/p53_direct.csv','datasets/myoglobin_direct.csv','datasets/ssym_direct.csv','datasets/S669_direct.csv']

#test_datasets = os.listdir(test_path)
#test_datasets = [ "S669_subsets/data/" + f for f in test_datasets]

datasets_path = '/orfeo/scratch/dssc/mceloria/PLM4Muts/'
#for td in test_datasets:
#    print(os.path.join(datasets_path, td))

preds = {n:[] for n in models} 

model_name  = args.model_name

print("AAA", lr, model_name)
model_class = globals()[model_name]
print(f'Training model {model_name}', flush=True) 
train_df = full_df
train_ds = ProteinDataset(train_df)
model = model_class()    
model.to(device) 
print("Debug A", flush=True)
if optimizer_name=="Adam":
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
if optimizer_name=="AdamW":
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

training_loader = DataLoader(train_ds, batch_size=1, num_workers = 0, shuffle = True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(training_loader), epochs=EPOCHS)
    
testing_dataframes = [None] * len(test_datasets)
testing_datasets   = [None] * len(test_datasets)
testing_loaders    = [None] * len(test_datasets)

train_loss   = np.zeros(EPOCHS) 
train_maes   = np.zeros(EPOCHS) 
val_losses   = [None] * len(test_datasets)
val_maes     = [None] * len(test_datasets)
val_corr     = [None] * len(test_datasets)
print("Debug B", flush=True) 
for test_no, test_dataset in enumerate(test_datasets):

    testing_dataframes[test_no] = pd.read_csv(os.path.join(datasets_path, test_dataset))
    testing_datasets[test_no]   = ProteinDataset(testing_dataframes[test_no])
    testing_loaders[test_no]    = DataLoader(testing_datasets[test_no], batch_size=1, num_workers = 0)
    val_losses[test_no] = np.zeros(EPOCHS)
    val_maes[test_no]   = np.zeros(EPOCHS)
    val_corr[test_no]   = np.zeros(EPOCHS)
print("Debug C", flush=True)    
for epoch in range(EPOCHS):
    labels, predictions =train(model, training_loader, device, optimizer, scheduler, epoch)
    L2loss = np.mean((np.array(labels) - np.array(predictions))**2)  
    MAE  = np.mean(np.abs(np.array(labels) - np.array(predictions)))
    RMSE = np.sqrt(L2loss)
    train_loss[epoch] = RMSE
    train_maes[epoch] = MAE
    print("-----------------------------------------------------------------------", flush=True) 
    print(f"Training Loss for epoch {epoch+1}/{EPOCHS} - {model_name}: RMSE[{train_loss[epoch]}] - MAE[{train_maes[epoch]}]", flush=True) 
    print("-----------------------------------------------------------------------", flush=True) 

    for test_no, testing_loader in enumerate(testing_loaders):
        labels, predictions = valid(model, testing_loader, device)
        L2loss = np.mean((np.array(labels) - np.array(predictions))**2) 
        MAE = np.mean(np.abs(np.array(labels) - np.array(predictions)))
        Correlation, p_value=pearsonr(labels, predictions)
        val_maes[test_no][epoch]=MAE
        val_corr[test_no][epoch]=Correlation
        RMSE=np.sqrt(L2loss)
        val_losses[test_no][epoch] = RMSE 
            
        print("***********************************************************************", flush=True) 
        print(f"Validation Dataset - {model_name}:\t{test_datasets[test_no]} len={len(testing_loader)}", flush=True) 
        print(f"Validation RMSE - {model_name}            for epoch {epoch+1}/{EPOCHS}:\t{val_losses[test_no][epoch]}", flush=True) 
        print(f"Validation MAE - {model_name}             for epoch {epoch+1}/{EPOCHS}:\t{val_maes[test_no][epoch]}", flush=True) 
        print(f"Validation Correlation - {model_name}     for epoch {epoch+1}/{EPOCHS}:\t{val_corr[test_no][epoch]}", flush=True) 
        print("***********************************************************************", flush=True) 
           
        if epoch==EPOCHS-1:
            for idx,(lab, pred) in enumerate(zip(labels,predictions)):
                if np.abs(lab-pred) > 0.0:
                    wild_seq = testing_dataframes[test_no].iloc[idx]['wild_type']
                    mut_seq  = testing_dataframes[test_no].iloc[idx]['mutated']
                    pos = testing_dataframes[test_no].iloc[idx]['pos']
                    ddg = testing_dataframes[test_no].iloc[idx]['ddg']
                    print(f"\n{test_datasets[test_no]}:\nwild_seq={wild_seq}\nmuta_seq={mut_seq}\npos={pos}\nlabels={lab}\tpredictions={pred}\n", 
                              flush=True)
         
model.to('cpu') 
        
#    torch.save(model.state_dict(), 'weights/' + model_name)
    
del model
torch.cuda.empty_cache()

print("Summary Direct Training", flush=True) 

test_dataset_names = [ s.rsplit('/', 1)[1].rsplit('.', 1)[0]  for s in test_datasets ] 

val_loss_names  = [ s + "_RMSE" for s in test_dataset_names]
val_mae_names   = [ s + "_MAE"  for s in test_dataset_names]
val_corr_names  = [ s + "_CORR"  for s in test_dataset_names]
train_loss_dict = {'train_RMSE': train_loss}

print(val_loss_names)
print(val_mae_names)
print(val_corr_names)

val_loss_df   = pd.DataFrame.from_dict(dict(zip(val_loss_names,  val_losses)))
val_mae_df    = pd.DataFrame.from_dict(dict(zip(val_mae_names,   val_maes)))
val_corr_df   = pd.DataFrame.from_dict(dict(zip(val_corr_names,  val_corr)))
train_loss_df = pd.DataFrame.from_dict(train_loss_dict)

train_loss_df["epoch"]= train_loss_df.index
val_corr_df["epoch"]  = val_corr_df.index
val_mae_df["epoch"]   = val_mae_df.index
val_loss_df["epoch"]  = val_loss_df.index

df = pd.concat([frame.set_index("epoch") for frame in [train_loss_df, val_loss_df, val_mae_df, val_corr_df]], axis=1, join="inner").reset_index()

print(df, flush=True) 

df.to_csv(f'../results3008/Epochs_Statistics_direcT_{model_name}_{training_name}_{optimizer_name}_{lr}_{EPOCHS}_L2.csv')

plt.figure(figsize=(8,6))
plt.plot(df["epoch"], df["train_RMSE"],            label="train_direct_RMSE",     color="black",      linestyle="-.")
plt.plot(df["epoch"], df["p53_direct_RMSE"],       label="p53_direct_RMSE",       color="tab:orange", linestyle="-")
plt.plot(df["epoch"], df["myoglobin_direct_RMSE"], label="myoglobin_direct_RMSE", color="tab:green",  linestyle="-")
plt.plot(df["epoch"], df["ssym_direct_RMSE"],      label="ssym_direct_RMSE",      color="tab:red",    linestyle="-")
plt.plot(df["epoch"], df["S669_direct_RMSE"],      label="S669_direct_RMSE",      color="tab:blue",   linestyle="-")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title(f"Model: {model_name}")
plt.legend()
plt.savefig(f'../results3008/Epochs_Loss_direcT_{model_name}_{training_name}_{optimizer_name}_{lr}_{EPOCHS}_L2.png')
plt.clf()

plt.figure(figsize=(8,6))
plt.plot(df["epoch"], df["p53_direct_MAE"],       label="p53_direct_MAE",       color="tab:orange")
plt.plot(df["epoch"], df["myoglobin_direct_MAE"], label="myoglobin_direct_MAE", color="tab:green")
plt.plot(df["epoch"], df["ssym_direct_MAE"],      label="ssym_direct_MAE",      color="tab:red")
plt.plot(df["epoch"], df["S669_direct_MAE"],      label="S669_direct_MAE",      color="tab:blue")
plt.xlabel("epoch")
plt.ylabel("MAE")
plt.title(f"Model: {model_name}")
plt.legend()
plt.savefig(f'../results3008/Epochs_MAE_direcT_{model_name}_{training_name}_{optimizer_name}_{lr}_{EPOCHS}_L2.png')
plt.clf()

plt.figure(figsize=(8,6))
plt.plot(df["epoch"], df["p53_direct_CORR"],       label="p53_direct_CORR",       color="tab:orange")
plt.plot(df["epoch"], df["myoglobin_direct_CORR"], label="myoglobin_direct_CORR", color="tab:green")
plt.plot(df["epoch"], df["ssym_direct_CORR"],      label="ssym_direct_CORR",      color="tab:red")
plt.plot(df["epoch"], df["S669_direct_CORR"],      label="S669_direct_CORR",      color="tab:blue")
plt.xlabel("epoch")
plt.ylabel("Pearson correlation coefficient")
plt.title(f"Model: {model_name}")
plt.legend()
plt.savefig(f'../results3008/Epochs_Pearsonr_direcT_{model_name}_{training_name}_{optimizer_name}_{lr}_{EPOCHS}_L2.png')
plt.clf()

