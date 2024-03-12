#import argparse
from Bio import SeqIO
#import csv
#import esm
import itertools
#import math
#import matplotlib.pyplot as plt
#from matplotlib import cm
#import numpy as np
import pandas as pd
import os
#import random
import re
import scipy
#from scipy import stats
#from scipy.stats import pearsonr
import string
#import time
import torch
import torch.distributed  as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
#import torch.nn.functional as F
#from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.cuda.amp import autocast
#import torch.multiprocessing as mp
from transformers import T5Tokenizer
#from typing import List, Tuple
#import warnings

torch.cuda.empty_cache()

#data-preprocessing step
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def remove_insertions(sequence: str) -> str:
    return sequence.translate(translation)

class ESM2_Dataset(Dataset):
    def __init__(self, df, name, max_length):
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
        return len(self.df)

    def __getitem__(self, idx):
        wild_seq = [self.df.iloc[idx]['wt_seq']]
        mut_seq  = [self.df.iloc[idx]['mut_seq']]
        pos = self.df.iloc[idx]['pos']
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        code = self.df.iloc[idx]['code']
        return (wild_seq, mut_seq, pos), ddg, code

class MSA_Dataset(Dataset):
    def __init__(self, df, name, dataset_dir, max_length, max_tokens):
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
#        self.didx = {}

    def read_msa(self, filename_wt, filename_mut):
        records_wt  = list(SeqIO.parse(filename_wt,  "fasta"))
        records_mut = list(SeqIO.parse(filename_mut, "fasta"))
        lseq = max([len(records_wt[i].seq) for i in range(len(records_wt))]) #lenght of longest seq of msa
        assert lseq < 1024
        nseqs = self.max_tokens//(lseq + 1)
        nseqs = min(int(nseqs), len(records_wt)) #select the numb of seq you are interested in
        idx = list(range(1, nseqs))
        pdb_list_wt  = [(records_wt[0].description, remove_insertions(str(records_wt[0].seq)))]#the first is included always
        pdb_list_mut = [(records_mut[0].description,remove_insertions(str(records_mut[0].seq)))]#the first is included always
        pdb_list_wt  = pdb_list_wt  + [(records_wt[i].description,  remove_insertions(str(records_wt[i].seq)))  for i in idx]
        pdb_list_mut = pdb_list_mut + [(records_mut[i].description, remove_insertions(str(records_mut[i].seq))) for i in idx]
        return pdb_list_wt, pdb_list_mut

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wild_seq = [self.df.iloc[idx]['wt_seq']]
        mut_seq  = [self.df.iloc[idx]['mut_seq']]
        wild_msa_path = self.df.iloc[idx]['wt_msa']
        mut_msa_path  = self.df.iloc[idx]['mut_msa']
        pos = self.df.iloc[idx]['pos']
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        wild_msa_filename = os.path.join(self.dataset_dir, wild_msa_path)
        mut_msa_filename  = os.path.join(self.dataset_dir, mut_msa_path )
        wild_msa, mut_msa = self.read_msa(wild_msa_filename, mut_msa_filename)
        code = self.df.iloc[idx]['code']
        return (wild_msa, mut_msa, pos), ddg, code

class ProstT5_Dataset(Dataset):
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
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

    def __getitem__(self, idx):
        seqs = [self.df.iloc[idx]['wt_seq'],   self.df.iloc[idx]['mut_seq'], 
                self.df.iloc[idx]['wt_struct'],self.df.iloc[idx]['mut_struct'] ]
        seqs_p  = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seqs]
        seqs_pp = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in seqs_p]
        seqs_e = self.tokenizer.batch_encode_plus(seqs_pp, add_special_tokens=True, padding="longest", return_tensors='pt')
        input_ids = seqs_e.input_ids
        attention_mask = seqs_e.attention_mask
        pos = self.df.iloc[idx]['pos']
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        code = self.df.iloc[idx]['code']
        return (input_ids, attention_mask, pos), ddg, code

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

