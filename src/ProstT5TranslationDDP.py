from argparser import *
import argparse
import csv
import itertools
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
import sys
import string
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.distributed  as dist
from torch.utils.data.distributed import DistributedSampler
import yaml
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple
import warnings
from torch.utils.data import Subset

def ddp_setup():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def from_cvs_files_in_dir_to_dfs_list(path):
    dir_path = path + "/data"
    datasets = os.listdir(dir_path)
    datasets_names = [ s.rsplit('.', 1)[0]  for s in datasets ]
    dfs = [None] * len(datasets)
    for i,d in enumerate(datasets):
        d_path = os.path.join(dir_path, d)
        dfs[i] = pd.read_csv(d_path, sep=',')
    return dfs, datasets_names

class ProteinDataset(Dataset):
    def __init__(self, df, name):
        self.name = name
        self.df = df
        self.tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        lengths = [len(s) for s in df['wt_seq'].to_list()]
        self.max_length = max(lengths) + 2

    def __getitem__(self, idx):
        seqs     = [self.df.iloc[idx]['wt_seq'], self.df.iloc[idx]['mut_seq']]
        
        wt_seq   = seqs[0]
        mut_seq  = seqs[1]
        ddg      = self.df.iloc[idx]['ddg']
        pdb_id   = self.df.iloc[idx]['pdb_id']
        mut_info = self.df.iloc[idx]['mut_info']
        pos  = self.df.iloc[idx]['pos']
        code = self.df.iloc[idx]['code']

        min_len = min([ len(s) for s in seqs])
        seqs_p  = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in seqs]
        seqs_p  = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in seqs_p]
        embeddings = self.tokenizer.batch_encode_plus(seqs_p, 
                                                      add_special_tokens=True, 
                                                      max_length=self.max_length,
                                                      padding="max_length", 
                                                      return_tensors='pt')
        return embeddings, (min_len, self.max_length), (wt_seq,mut_seq,ddg,pdb_id,mut_info,pos,code)

    def __len__(self):
        return len(self.df)

class OutputDatatype():
    def __init__(self, wt_seq, wt_struct, mut_seq, mut_struct, ddg, pdb_id, mut_info, pos, code): 
        # wt_seq,mut_seq,ddg,pdb_id,mut_info,pos,code
        self.wt_seq = wt_seq
        self.wt_struct = wt_struct
        self.mut_seq = mut_seq
        self.mut_struct = mut_struct
        self.ddg = ddg
        self.pdb_id = pdb_id
        self.mut_info = mut_info
        self.pos = pos
        self.code = code

    def to_dict(self):
        return {
            'wt_seq': self.wt_seq,
            'wt_struct': self.wt_struct,
            'mut_seq': self.mut_seq,
            'mut_struct': self.mut_struct,
            'ddg': self.ddg,
            'pdb_id': self.pdb_id,
            'mut_info': self.mut_info,
            'pos' : self.pos,
            'code': self.code,
        }

def translate(dataloader, model, local_rank):
    gen_kwargs_aa2fold = {
                  "do_sample": True,
                  "num_beams": 3,
                  "top_p" : 0.95,
                  "temperature" : 1.2,
                  "top_k" : 6,
                  "repetition_penalty" : 1.2,
                  }
    results = []
    global_rank = dist.get_rank()
    workers = 0
    batch_size = 1
    world_size = dist.get_world_size()
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
    def run_translate(loader, base_progress=0):
        len_loader = len(loader)
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                idx = base_progress + idx
                embeddings, (min_len, max_len), (wt_seq,mut_seq,ddg,pdb_id,mut_info,pos,code) = batch
                embeddings = embeddings.to(local_rank)
                print(f"{dataloader.name}\ton GPU {global_rank}\tbatch_idx:{idx+1}/{len_loader}: {code}", 
                      flush=True)
                print(embeddings.input_ids.shape, embeddings.attention_mask.shape, max_len, min_len)
                translations = model.generate(
                                     input_ids=embeddings.input_ids.squeeze(0),
                                     attention_mask=embeddings.attention_mask.squeeze(0),
                                     max_length=int(max_len), # max length of generated text
                                     min_length=int(min_len), # minimum length of the generated text
                                     early_stopping=True, # stop early if end-of-text token is generated
                                     #num_return_sequences=1, # return only a single sequence
                                     **gen_kwargs_aa2fold
                                     )
                # Decode and remove white-spaces between tokens
                decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
                structures = [ "".join(ts.split(" ")) for ts in decoded_translations ]
                wt_struct  = structures[0]
                mut_struct = structures[1]
                row=OutputDatatype(wt_seq[0],wt_struct,mut_seq[0],mut_struct,ddg.item(),pdb_id[0],mut_info[0],pos[0].item(),code[0])
                if (base_progress!=0 and global_rank==0) or base_progress==0:
                    results.append(row)


    # switch to evaluate mode
    model.eval()
    run_translate(dataloader.dataloader)
    dist.barrier()
    if (len(dataloader.dataloader.sampler) * world_size < len(dataloader.dataloader.dataset)):
        aux_dataset = Subset(dataloader.dataloader.dataset,
                             range(len(dataloader.dataloader.sampler) * world_size, 
                             len(dataloader.dataloader.dataset)))
        aux_dataloader = torch.utils.data.DataLoader(aux_dataset, 
                                                     batch_size=batch_size, 
                                                     shuffle=False,
                                                     num_workers=workers, 
                                                     pin_memory=False)
        print("RUG", len(dataloader.dataloader))
        run_translate(aux_dataloader, len(dataloader.dataloader))
        dist.barrier()
    return results

class ProteinDataLoader():
    def __init__(self, dataset, batch_size, num_workers, shuffle, pin_memory, sampler):
        self.name = dataset.name
        self.df = dataset.df
        self.dataloader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=shuffle,
                                     pin_memory=pin_memory,
                                     sampler=sampler)


def main(input_file, output_file):
    ddp_setup()
    local_rank  = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = dist.get_world_size()
    df     = pd.read_csv(input_file, sep=',')
    print("DEBUG", input_file.rsplit('/', 1))
    in_dir       = input_file.rsplit('/', 1)[0]
    infile_name  = input_file.rsplit('/', 1)[1]
    out_dir      = output_file.rsplit('/', 1)[0]
    outfile_name = output_file.rsplit('/', 1)[1]
    ds = ProteinDataset(df, infile_name)
    Dsampler = DistributedSampler(ds,shuffle=False,drop_last=True)
    dl = ProteinDataLoader(ds, batch_size=1, num_workers=0, shuffle=False, pin_memory=False, sampler=Dsampler)

    model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5")
    # only GPUs support half-precision currently; if you want to run on CPU use full-precision 
    model.to(local_rank)
    result=translate(dl, model, local_rank)
    dist.barrier()
    df=pd.DataFrame.from_records([r.to_dict() for r in result])
    tmp_filenames = [out_dir + str(f"/tmp_translate.{i}.csv") for i in range(world_size)]
    df.to_csv(tmp_filenames[global_rank], index=False)
    if global_rank==0:
        dfs=[None]*world_size
        for i in range(world_size):
            dfs[i]=pd.read_csv(tmp_filenames[i])
        res_df = pd.concat(dfs)
        res_df.columns = ['wt_seq','wt_struct','mut_seq','mut_struct','ddg','pdb_id','mut_info','pos','code']
        res_df = res_df.sort_values(by=['code'])
        res_df.to_csv(output_file, index=False)
        for i in range(world_size):
            if os.path.exists(tmp_filenames[i]):
                os.remove(tmp_filenames[i])

    #tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
    #with torch.no_grad():
    #    for idx, batch in enumerate(dl.dataloader):
    #        seq_e, (min_len, max_len), pos, seq, code = batch
    #        seq_e = seq_e.to(local_rank)
    #        print(f"{dl.name}\ton GPU {global_rank}\tbatch_idx:{idx+1}/{len_dataloader}", flush=True)
    #        print(seq_e.input_ids.shape, seq_e.attention_mask.shape, max_len, min_len)
    #        translations = model.generate( 
    #              seq_e.input_ids.squeeze(0), 
    #              attention_mask=seq_e.attention_mask.squeeze(0), 
    #              max_length=int(max_len), # max length of generated text
    #              min_length=int(min_len), # minimum length of the generated text
    #              early_stopping=True, # stop early if end-of-text token is generated
    #              #num_return_sequences=1, # return only a single sequence
    #              **gen_kwargs_aa2fold
    #        )
    #        # Decode and remove white-spaces between tokens
    #        decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    #        structures = [ "".join(ts.split(" ")) for ts in decoded_translations ]
    #        print()
            
            # predicted 3Di strings
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    # Define the args from argparser
    args = argparser_translator()
    input_file  = args.input_file
    output_file = args.output_file
    main(input_file, output_file)
