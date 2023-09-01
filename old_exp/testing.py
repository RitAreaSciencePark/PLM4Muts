#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
import re
import matplotlib.pyplot as plt
torch.cuda.empty_cache()

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0")


# In[2]:


class ProstT5_mut(nn.Module):

    def __init__(self):
        super().__init__() 
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5") 
        self.classifier = nn.Linear(2048, 1)
        self.const1 = torch.nn.Parameter(torch.ones((1,2048)))
        self.const2 = torch.nn.Parameter(-1 * torch.ones((1,2048)))
        

    def forward(self, token_ids1, token_ids2, pos): 

        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state  
        tmp1 = outputs1[:,pos+1,:]
        merged_out1 = tmp1.reshape(1,-1)

        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp2 = outputs2[:,pos+1,:]
        merged_out2 = tmp2.reshape(1,-1)
        
        outputs = self.const1 * merged_out1+ self.const2 * merged_out2

        logits = self.classifier(outputs)
        logits = logits.unsqueeze(0)
        
        return logits


# In[3]:


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


# In[4]:


def valid(model, testing_loader):
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels, eval_scores = [], [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            input_ids1, input_ids2, pos, labels = batch   
            input_ids1 = input_ids1['input_ids'].to(device)[0] 
            input_ids2 = input_ids2['input_ids'].to(device)[0] 
  
            labels = labels.to(device)
            logits = model(token_ids1 = input_ids1, token_ids2 = input_ids2, pos = pos)     

            loss = torch.nn.functional.mse_loss(logits, labels)
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
                     
            eval_labels.extend(labels.cpu().detach())
            eval_preds.extend(logits.cpu().detach())
            
  
    labels = [id.item() for id in eval_labels]
    predictions = [id.item() for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    print("Validation Loss:", {eval_loss})

    return labels, predictions


# In[5]:


model = ProstT5_mut()
model.load_state_dict(torch.load('/orfeo/scratch/area/cuturellof/ProstT5/weights/ProstT5_mut'))
model.to(device)


# In[8]:


test_datasets = ['S669.csv']
datasets_path = '/orfeo/scratch/area/cuturellof/mutations_experiments/ProstT5/'
for test_no, test_dataset in enumerate(test_datasets):

    test_df = pd.read_csv(os.path.join(datasets_path, test_dataset))
    test_ds = ProteinDataset(test_df)
    testing_loader = DataLoader(test_ds, batch_size=1, num_workers = 2)
    labels, predictions = valid(model, testing_loader)

    print('MAE', {np.mean(np.abs(np.array(labels) - np.array(predictions)))}, 'Correlation', {pearsonr(labels, predictions)})     


# In[9]:


plt.scatter(labels,predictions)
plt.show()


# In[ ]:


plt.scatter(labels,predictions)
plt.show()

