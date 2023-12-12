import pandas as pd
import os
import torch

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

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

