from transformers import T5Tokenizer, T5EncoderModel
import re

import pandas as pd
import numpy as np
import Bio
from Bio import SeqIO
import os
import torch
import math
#import esm

from torch import nn
from torch.utils.data import Dataset, DataLoader
import scipy
from scipy import stats
import torch.nn.functional as F
from torch.cuda.amp import autocast
torch.cuda.empty_cache()

import warnings
warnings.filterwarnings("ignore")

HIDDEN_UNITS_POS_CONTACT = 5
#device = torch.device("cpu")
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

class ProstT5_mut(nn.Module):

    def __init__(self):
        super().__init__()
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(1024, 1)
        self.const1 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const2 = torch.nn.Parameter(-1 * torch.ones((1,1024)))


    def forward(self, token_ids1, token_ids2, pos):
      #  with torch.no_grad():
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        outputs = self.const1 * outputs1[:,pos + 1,:] + self.const2 * outputs2[:,pos + 1,:]
        logits = self.classifier(outputs)
        return logits


class ProteinDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

    def __getitem__(self, idx):
        wild_seq = [self.df.iloc[idx]['wild_type'], self.df.iloc[idx]['structure']]
        wild_seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in wild_seq]
        wild_seq = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in wild_seq]
        prostt5_batch_tokens1 = self.tokenizer.batch_encode_plus(wild_seq, add_special_tokens=True, padding="longest", return_tensors='pt')
        
        mut_seq = [self.df.iloc[idx]['mutated'], self.df.iloc[idx]['structure']]
        mut_seq = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in mut_seq]
        mut_seq = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in mut_seq]
        prostt5_batch_tokens2 = self.tokenizer.batch_encode_plus(mut_seq, add_special_tokens=True, padding="longest", return_tensors='pt')
        
        pos = self.df.iloc[idx]['pos']
        return prostt5_batch_tokens1, prostt5_batch_tokens2, pos, torch.unsqueeze(torch.FloatTensor([self.df.iloc[idx]['ddg']]), 0)

    def __len__(self):
        return len(self.df)


def train(epoch):
#    scaler = torch.cuda.amp.GradScaler()
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.train()

    for idx, batch in enumerate(training_loader):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f"\tidx={idx}/{len(training_loader)}")
        print(f"\ttot={t};res={r};all={a};free={f}\n")
        input_ids1, input_ids2, pos, labels = batch 
        input_ids1 = input_ids1['input_ids'].to(device)[0] 
        input_ids2 = input_ids2['input_ids'].to(device)[0] 
        labels = labels.to(device)
        pos = pos.to(device)
        
#        with autocast(dtype=torch.float16):
            
        logits = model(token_ids1 = input_ids1, token_ids2 = input_ids2, pos = pos).to(device) 
        loss = torch.nn.functional.mse_loss(logits, labels)

        tr_loss += loss.item()
        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        scaler.scale(loss).backward()
#        scaler.unscale_(optimizer)
#        scaler.step(optimizer)
        scheduler.step()
#        scaler.update()
        
    epoch_loss = tr_loss / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")


lr = 1e-5
EPOCHS = 3

models = ['ProstT5_mut']

full_df = pd.read_csv('training_set.csv',sep=',')

preds = {n:[] for n in models} 
true = [None] 

for model_name in models:
    model_class = globals()[model_name]
    print(f'Training model {model_name}')
    train_df = full_df
    train_ds = ProteinDataset(train_df)
        
    model = model_class()    
    model.to(device)
#    model.full() if device=='cpu' else model.half()
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    training_loader = DataLoader(train_ds, batch_size=1, num_workers = 2, shuffle = True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(training_loader), epochs=EPOCHS)
        
    for epoch in range(EPOCHS):
        print(epoch)
        train(epoch)
         
    model.to('cpu') 
        
    torch.save(model.state_dict(), 'weights/' + model_name)
    
    del model


