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
import torch.nn.functional as F
from transformers import T5Tokenizer
from typing import List, Tuple
import warnings

torch.cuda.empty_cache()


class Tester:
    def __init__(
        self,
        output_dir: str,
    ) -> None:
        self.output_dir  = output_dir
        self.local_rank  = 0
        self.epoch = 0
        self.lr = 0

    def initialize_files(self):
        self.result_dir    = self.output_dir + "/results"
        if not(os.path.exists(self.output_dir) and os.path.isdir(self.output_dir)):
            os.makedirs(self.output_dir)
        if not(os.path.exists(self.result_dir) and os.path.isdir(self.result_dir)):
            os.makedirs(self.result_dir)

        self.test_resfiles  = [self.result_dir + f"/{test.name}_test.res" for test in self.test_dls]

        for test_resfile in self.test_resfiles:
            with open(test_resfile, "w") as t_log:
                t_log.write("epoch,rmse,mae,corr\n")

    def _load_snapshot(self, model, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        #model.module.load_state_dict(snapshot["MODEL_STATE"])
        model.load_state_dict(snapshot["MODEL_STATE"])
        self.epoch = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epoch+1}", flush=True)


    def test(self, test_model, test_dls, snapshot_file):
        self.test_model_name = test_model.name
        self.test_model = test_model.to(self.local_rank)
        #self.test_model = DDP(self.test_model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.test_dls = test_dls
        self.initialize_files()
        self.test_rmses   = torch.zeros(len(self.test_dls))
        self.test_maes    = torch.zeros(len(self.test_dls))
        self.test_corrs   = torch.zeros(len(self.test_dls))
        if os.path.exists(snapshot_file):
            print("Loading snapshot", flush=True)
            self._load_snapshot(test_model, snapshot_file)
        #print("DEBUG1", test_model.state_dict()["const1"])
        #print("DEBUG2", test_model.state_dict()["const2"])
        #print("DEBUG3", test_model.state_dict()["const3"])
        #print("DEBUG4", test_model.state_dict()["const4"])
        self.test_model.eval()
        for test_no, test_dl in enumerate(self.test_dls):
            self.t_preds, self.t_labels = [], []
            len_dataloader = len(test_dl.dataloader)
            diff_file   =  self.result_dir + f"/{test_dl.name}_labels_preds.diffs"
            with open(diff_file, "w") as t_diffs:
                t_diffs.write(f"code,pos,ddg,pred\n")
            with torch.no_grad():
                for idx, batch in enumerate(test_dl.dataloader):
                    print(f"dataset:{test_dl.name}\tGPU:{self.local_rank}\tbatch_idx:{idx+1}/{len_dataloader}\ttest:{batch[-1]}", flush=True)
                    testX, testY, code = batch
                    testYhat = self.test_model(*testX, self.local_rank).to(self.local_rank)
                    testY_cpu = testY.cpu().detach()
                    testYhat_cpu = testYhat.cpu().detach()
                    pos = testX[-1]
                    pos_cpu = pos.cpu().detach().item()
                    with open(diff_file, "a") as t_diffs:
                       t_diffs.write(f"{code},{pos_cpu},{testY_cpu.item()},{testYhat_cpu.item()}\n")
                    self.t_labels.extend(testY_cpu)
                    self.t_preds.extend(testYhat_cpu)
            l_t_labels = torch.tensor(self.t_labels).to("cpu")
            l_t_preds  = torch.tensor(self.t_preds).to("cpu")
            l_t_mse  = torch.mean(         (l_t_labels - l_t_preds)**2)
            l_t_mae  = torch.mean(torch.abs(l_t_labels - l_t_preds)   )
            l_t_rmse = torch.sqrt(l_t_mse)
            l_t_corr, _ = pearsonr(l_t_labels.tolist(), l_t_preds.tolist())
            self.test_maes[test_no]  = l_t_mae
            self.test_corrs[test_no] = l_t_corr
            self.test_rmses[test_no] = l_t_rmse
            print(f"{test_dl.name}: on GPU {self.local_rank} test\t"
                      f"rmse = {self.test_rmses[test_no]}\t"
                      f"mae = {self.test_maes[test_no]}\t"
                      f"corr = {self.test_corrs[test_no]}")

            with open(self.result_dir + f"/{test_dl.name}_test.res", "a") as t_res:
                        t_res.write(f"{self.epoch+1},{self.test_rmses[test_no]},{self.test_maes[test_no]},{self.test_corrs[test_no]}\n")





