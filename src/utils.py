import pandas as pd
import os
import torch

def ddp_setup():
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def from_cvs_files_in_dir_to_dfs_list(path):
    dir_path = path + "/MSA_3Di_databases"
    datasets = os.listdir(dir_path)
    #datasets_names = [ s.rsplit('/', 1)[1].rsplit('.', 1)[0]  for s in datasets ]
    datasets_names = [ s.rsplit('.', 1)[0]  for s in datasets ]
    dfs = [None] * len(datasets)
    for i,d in enumerate(datasets):
        d_path = os.path.join(dir_path, d)
        dfs[i] = pd.read_csv(d_path, sep=',')
    return dfs, datasets_names

