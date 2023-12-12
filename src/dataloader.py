import argparse
from Bio import SeqIO
import csv
import esm
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import random
import re
import scipy
from scipy import stats
from scipy.stats import pearsonr
import string
import time
import torch
import torch.distributed  as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from transformers import T5Tokenizer
from typing import List, Tuple
import warnings

torch.cuda.empty_cache()
#warnings.filterwarnings("ignore")

#data-preprocessing step
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def remove_insertions(sequence: str) -> str:
    return sequence.translate(translation)

def read(filename_wt, filename_mut, max_tokens):
    records_wt  = list(SeqIO.parse(filename_wt,  "fasta"))
    records_mut = list(SeqIO.parse(filename_mut, "fasta"))
    lseq = max([len(records_wt[i].seq) for i in range(len(records_wt))]) #lenght of longest seq of msa
    assert lseq < 1024
    #if lseq > 1024:
    #    for seq in range(len(records)):
    #        records[seq] = records[seq][:1023]
    #        lseq = 1023
#    max_tokens = 15000
    #if(nseqs * lseq > (100 * 100)):
    #    nseqs = (100 * 100)//(lseq + 1)
    #if nseqs * lseq > max_tokens:
    nseqs = max_tokens//(lseq + 1)
    nseqs = min(int(nseqs), len(records_wt)) #select the numb of seq you are interested in
    
    #idx = random.sample(list(range(1, len(records_wt))), nseqs - 1) #extract nseq-2 idx
    idx = list(range(1, nseqs))
    #idxs = []
    #for i in range(nseq - 1):
    #    idxs.append(records[i].id)
    #for i in idx:
    #    idxs.append(records[i].id)
    #return [(records[i].description, remove_insertions(str(records[i].seq))) for i in range(nseqs)], nseqs, idxs
    pdb_list_wt  = [(records_wt[0].description,  remove_insertions(str(records_wt[0].seq)))] #the first is included always
    pdb_list_mut = [(records_mut[0].description, remove_insertions(str(records_mut[0].seq)))] #the first is included always
    pdb_list_wt  = pdb_list_wt  + [(records_wt[i].description,  remove_insertions(str(records_wt[i].seq)))  for i in idx]
    pdb_list_mut = pdb_list_mut + [(records_mut[i].description, remove_insertions(str(records_mut[i].seq))) for i in idx]
    return pdb_list_wt, pdb_list_mut
    #return pdb_list + [(records[idx[i]].description, remove_insertions(str(records[idx[i]].seq))) for i in range(1, nseqs - 1)], nseqs, idxs


class ESM_Dataset(Dataset):
    def __init__(self, df, name, max_length):
        ''' 
        Dataset class.
        Args:
            self.dir: directory where the files are found
            nseq: how many sequences for each MSA are selected
            filenames: list of all files located in the directory
        '''
        self.name = name
        self.df = df
        wt_lengths  = [len(s) for s in df['wt_seq'].to_list()] 
        mut_lengths = [len(s) for s in df['mut_seq'].to_list()]
        self.df["wt_len_seq"]  = wt_lengths
        self.df["mut_len_seq"] = mut_lengths
        self.max_length = max_length
        self.df = self.df.drop(self.df[self.df.wt_len_seq > self.max_length - 2].index)
        self.df = self.df.drop(self.df[self.df.mut_len_seq > self.max_length - 2].index)

    def __len__(self):
        ''' How many files are present in the directory. '''
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Overiding of index operator in order to select a given file from the directory.
        Args:
        ''' 
        wild_seq = [self.df.iloc[idx]['wt_seq']]
        mut_seq  = [self.df.iloc[idx]['mut_seq']]
        
        pos = self.df.iloc[idx]['pos']
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        
        #ddg = torch.unsqueeze(ddg, 0)
        code = self.df.iloc[idx]['code']
        return (wild_seq, mut_seq, pos), ddg, code

class MSA_Dataset(Dataset):
    def __init__(self, df, name, dataset_dir, max_length, max_tokens):
        ''' 
        Dataset class.
        Args:
            self.dir: directory where the files are found
            nseq: how many sequences for each MSA are selected
            filenames: list of all files located in the directory
        '''
        self.name = name
        self.df = df
        wt_lengths  = [len(s) for s in df['wt_seq'].to_list()] 
        mut_lengths = [len(s) for s in df['mut_seq'].to_list()]
        self.df["wt_len_seq"]=wt_lengths
        self.df["mut_len_seq"]=mut_lengths
        self.max_length = max_length
        self.df = self.df.drop(self.df[self.df.wt_len_seq  > self.max_length - 2].index)
        self.df = self.df.drop(self.df[self.df.mut_len_seq > self.max_length - 2].index)
        self.dataset_dir = dataset_dir
        self.max_tokens = max_tokens
        self.didx = {}

    def read_msa(self, filename_wt, filename_mut):
        records_wt  = list(SeqIO.parse(filename_wt,  "fasta"))
        records_mut = list(SeqIO.parse(filename_mut, "fasta"))
        lseq = max([len(records_wt[i].seq) for i in range(len(records_wt))]) #lenght of longest seq of msa
        assert lseq < 1024
        nseqs = self.max_tokens//(lseq + 1)
        nseqs = min(int(nseqs), len(records_wt)) #select the numb of seq you are interested in
        #idx = random.sample(list(range(1, len(records_wt))), nseqs - 1) #extract nseq-1 idx
        #self.didx.setdefault((len(records_wt),nseqs), random.sample(list(range(1, len(records_wt))), nseqs - 1)) 
        #idx = self.didx[(len(records_wt),nseqs)]
        idx = list(range(1, nseqs))
        pdb_list_wt  = [(records_wt[0].description,  remove_insertions(str(records_wt[0].seq)))] #the first is included always
        pdb_list_mut = [(records_mut[0].description, remove_insertions(str(records_mut[0].seq)))] #the first is included always
        pdb_list_wt  = pdb_list_wt  + [(records_wt[i].description,  remove_insertions(str(records_wt[i].seq)))  for i in idx]
        pdb_list_mut = pdb_list_mut + [(records_mut[i].description, remove_insertions(str(records_mut[i].seq))) for i in idx]
        return pdb_list_wt, pdb_list_mut

    def __len__(self):
        ''' How many files are present in the directory. '''
        return len(self.df)

    def __getitem__(self, idx):
        '''
        Overiding of index operator in order to select a given file from the directory.
        Args:
            msa_name: name of the file 
            msa: selected msa  
            n_seq: how many sequences have been selected 
        ''' 
        wild_seq = [self.df.iloc[idx]['wt_seq']]
        mut_seq  = [self.df.iloc[idx]['mut_seq']]
        
        wild_msa_path = self.df.iloc[idx]['wt_msa']
        mut_msa_path  = self.df.iloc[idx]['mut_msa']
        
        pos = self.df.iloc[idx]['pos']
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        
        wild_msa_filename = os.path.join(self.dataset_dir, wild_msa_path)
        mut_msa_filename  = os.path.join(self.dataset_dir, mut_msa_path )
        wild_msa, mut_msa = self.read_msa(wild_msa_filename, mut_msa_filename)
        #mut_msa  = read_msa(mut_msa_filename,  nseqs = 1)
        #ddg = torch.unsqueeze(ddg, 0)
        code = self.df.iloc[idx]['code']
        return (wild_msa, mut_msa, pos), ddg, code

class ProteinDataset(Dataset):
    def __init__(self, df, name, max_length):
        self.name = name
        self.df = df
        wt_lengths  = [len(s) for s in df['wt_seq'].to_list()] 
        mut_lengths = [len(s) for s in df['mut_seq'].to_list()]
        self.df["wt_len_seq"]=wt_lengths
        self.df["mut_len_seq"]=mut_lengths
        if max_length:
            self.max_length = max_length
            self.df = self.df.drop(self.df[self.df.wt_len_seq >  self.max_length - 2].index)
            self.df = self.df.drop(self.df[self.df.mut_len_seq > self.max_length - 2].index)
        else:
            self.max_length = max(wt_lengths)
        print("DEBUG: max_length =", self.max_length)
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

    def __getitem__(self, idx):
        wild_seq    = [self.df.iloc[idx]['wt_seq']]
        mut_seq     = [self.df.iloc[idx]['mut_seq']]
        wild_struct = [self.df.iloc[idx]['wt_struct']]
        mut_struct  = [self.df.iloc[idx]['mut_struct']]
        wild_seq_p  = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in wild_seq]
        mut_seq_p   = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in mut_seq]
        wild_struct_p = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in wild_struct]
        mut_struct_p  = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in mut_struct]
        wild_seq_p    = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in wild_seq_p]
        mut_seq_p     = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in mut_seq_p]
        mut_struct_p  = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in wild_struct_p]
        wild_struct_p = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in mut_struct_p]
        wild_seq_e = self.tokenizer.batch_encode_plus(wild_seq_p, add_special_tokens=True, max_length=self.max_length,
                                                      padding="max_length", return_tensors='pt')
        mut_seq_e  = self.tokenizer.batch_encode_plus(mut_seq_p,  add_special_tokens=True, max_length=self.max_length, 
                                                      padding="max_length", return_tensors='pt')
        wild_struct_e = self.tokenizer.batch_encode_plus(wild_struct_p, add_special_tokens=True, max_length=self.max_length,
                                                      padding="max_length", return_tensors='pt')
        mut_struct_e = self.tokenizer.batch_encode_plus(mut_struct_p, add_special_tokens=True, max_length=self.max_length,
                                                      padding="max_length", return_tensors='pt')
        pos = self.df.iloc[idx]['pos']
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        code = self.df.iloc[idx]['code']
        #ddg = torch.unsqueeze(ddg, 0)
        return (wild_seq_e, mut_seq_e, wild_struct_e, mut_struct_e, pos), ddg, code

    def __len__(self):
        return len(self.df)

def custom_collate(batch):
    assert len(batch)==1
    (wild, mut, pos), ddg, code = batch[0]
    pos = torch.tensor([pos])
    output = ([wild], [mut], pos), ddg.reshape((-1,1)), code
    return output

class ProteinDataLoader():
    def __init__(self, dataset, batch_size, num_workers, shuffle, pin_memory, sampler, custom_collate_fn=None):
        self.name = dataset.name
        self.df = dataset.df
        self.dataloader = DataLoader(dataset, 
                                     batch_size=batch_size, 
                                     num_workers=num_workers, 
                                     shuffle=shuffle, 
                                     pin_memory=pin_memory, 
                                     sampler=sampler,
                                     collate_fn=custom_collate_fn)
