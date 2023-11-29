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

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str,str]]:
    records = list(SeqIO.parse(filename, "fasta"))
    
    lseq = max([len(records[i].seq) for i in range(len(records))]) #lenght of longest seq of msa
    if lseq > 1024:
        for seq in range(len(records)):
            records[seq] = records[seq][:1023]
            lseq = 1023

    nseq = min(int(nseq), len(records)) #select the numb of seq you are interested in
    if(nseq * lseq > (300 * 100)):
        nseq = (300 * 100)//(lseq + 1)
    
    idx  = random.sample(list(range(0, len(records) - 1)), nseq - 1) #extract nseq-1 idx
    idxs = []
    #for i in range(nseq - 1):
    #    idxs.append(records[i].id)
    for i in idx:
        idxs.append(records[i].id)
    pdb_list = [(records[0].description, remove_insertions(str(records[0].seq)))] #the first is included always
    return pdb_list + [(records[i].description, remove_insertions(str(records[i].seq))) for i in range(1, nseq - 1)], nseq, idxs

class MSA_Dataset(Dataset):
    def __init__(self, df, name, dataset_dir, nseq):
        ''' 
        Dataset class.
        Args:
            self.dir: directory where the files are found
            nseq: how many sequences for each MSA are selected
            filenames: list of all files located in the directory
        '''
        self.df = df
        self.name = name
        self.dataset_dir = dataset_dir
        self.nseq = nseq

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
        wild_seq = [self.df.iloc[idx]['wild_type']]
        mut_seq  = [self.df.iloc[idx]['mutated']]
        
        wild_msa_path = self.df.iloc[idx]['file_msa_wild']
        mut_msa_path  = self.df.iloc[idx]['file_msa_mutated']
        
        pos = self.df.iloc[idx]['pos']
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        
        wild_msa_filename = os.path.join(self.dataset_dir, wild_msa_path)
        mut_msa_filename  = os.path.join(self.dataset_dir, mut_msa_path )
        wild_msa, wild_n_seq, wild_idxs = read_msa(wild_msa_filename, nseq = self.nseq)
        mut_msa,  mut_n_seq,  mut_idxs  = read_msa(mut_msa_filename,  nseq = self.nseq)
        #ddg = torch.unsqueeze(ddg, 0)
        return (wild_msa, mut_msa, pos), ddg, (wild_seq[0], mut_seq[0])

class ProteinDataset(Dataset):
    def __init__(self, df, name):
        self.name = name
        self.df = df
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        lengths = [len(s) for s in df['wild_type'].to_list()]
        self.max_length = max(lengths) + 2

    def __getitem__(self, idx):
        wild_seq = [self.df.iloc[idx]['wild_type']]
        mut_seq  = [self.df.iloc[idx]['mutated']]
        struct   = [self.df.iloc[idx]['structure']]
        wild_seq_p = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in wild_seq]
        mut_seq_p  = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in mut_seq]
        struct_p   = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in struct]
        wild_seq_p = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in wild_seq_p]
        mut_seq_p  = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in mut_seq_p]
        struct_p   = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in struct_p]
        wild_seq_e = self.tokenizer.batch_encode_plus(wild_seq_p, add_special_tokens=True, max_length=self.max_length,
                                                      padding="max_length", return_tensors='pt')
        mut_seq_e  = self.tokenizer.batch_encode_plus(mut_seq_p,  add_special_tokens=True, max_length=self.max_length, 
                                                      padding="max_length", return_tensors='pt')
        struct_e   = self.tokenizer.batch_encode_plus(struct_p,   add_special_tokens=True, max_length=self.max_length,
                                                      padding="max_length", return_tensors='pt')
        pos = self.df.iloc[idx]['pos']
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        #ddg = torch.unsqueeze(ddg, 0)
        return (wild_seq_e, mut_seq_e, struct_e, pos), ddg, (wild_seq[0], mut_seq[0])

    def __len__(self):
        return len(self.df)



def custom_collate(batch):
    assert len(batch)==1
    (wild, mut, pos), ddg, (wild_seq, mut_seq) = batch[0]
    pos = torch.tensor([pos])
    return ([wild], [mut], pos), ddg.reshape((-1,1)), (wild_seq, mut_seq)


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


