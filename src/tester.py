# SPDX-FileCopyrightText: 2024 (C) 2024 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import re
import scipy
from scipy import stats
from scipy.stats import pearsonr
import string
import torch
import torch.nn.functional as F

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
                t_log.write("epoch,rmse,mae,corr,p-value\n")

    def _load_snapshot(self, model, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.epoch = snapshot["EPOCHS_RUN"]
        print(f"Resuming the training from snapshot at Epoch {self.epoch}", flush=True)
        model.load_state_dict(snapshot["MODEL_STATE"])


    def test(self, test_model, test_dls, snapshot_file):
        self.test_model_name = test_model.name
        self.test_model = test_model.to(self.local_rank)
        self.test_dls = test_dls
        self.initialize_files()
        self.test_rmses   = torch.zeros(len(self.test_dls))
        self.test_maes    = torch.zeros(len(self.test_dls))
        self.test_corrs   = torch.zeros(len(self.test_dls))
        self.test_pvalue  = torch.zeros(len(self.test_dls))
        if os.path.exists(snapshot_file):
            print("Loading snapshot", flush=True)
            self._load_snapshot(test_model, snapshot_file)
        self.test_model.eval()
        for test_no, test_dl in enumerate(self.test_dls):
            len_dataloader = len(test_dl.dataloader)
            local_t_preds, local_t_labels = torch.zeros(len_dataloader), torch.zeros(len_dataloader)
            diff_file   =  self.result_dir + f"/{test_dl.name}_labels_preds.diffs"
            with open(diff_file, "w") as t_diffs:
                t_diffs.write(f"code,pos,ddg,pred\n")
            with torch.no_grad():
                for idx, batch in enumerate(test_dl.dataloader):
                    testX, testY, code = batch
                    code = "".join(code)
                    print(f"dataset:{test_dl.name}\tGPU:{self.local_rank}\tbatch_idx:{idx+1}/{len_dataloader}\ttest:{code}", flush=True)
                    first_cpu, second_cpu, pos_cpu = self.test_model.preprocess(*testX)
                    first_gpu  =  first_cpu.to(self.local_rank)
                    second_gpu = second_cpu.to(self.local_rank)
                    pos_gpu    =    pos_cpu.to(self.local_rank)

                    testYhat_gpu = self.test_model(first_gpu, second_gpu, pos_gpu)
                    testY_cpu    =        testY.detach().to("cpu").item()
                    testYhat_cpu = testYhat_gpu.detach().to("cpu").item()
                    pos_cpu      =      pos_gpu.detach().to("cpu").item()
                    local_t_preds[idx]  = testYhat_cpu
                    local_t_labels[idx] = testY_cpu
                    if test_dl.inference:
                        s = f"{code},{pos_cpu},,{testYhat_cpu}\n"
                    else:
                        s = f"{code},{pos_cpu},{testY_cpu},{testYhat_cpu}\n"
                    with open(diff_file, "a") as t_diffs:
                        t_diffs.write(s)
            if not test_dl.inference:
                l_t_mse  = torch.mean(         (local_t_labels - local_t_preds)**2)
                l_t_mae  = torch.mean(torch.abs(local_t_labels - local_t_preds)   )
                l_t_rmse = torch.sqrt(l_t_mse)
                l_t_corr, pvalue = pearsonr(local_t_labels.tolist(), local_t_preds.tolist())
                self.test_maes[test_no]  = l_t_mae
                self.test_corrs[test_no] = l_t_corr
                self.test_rmses[test_no] = l_t_rmse
                self.test_pvalue[test_no]= pvalue
                print(f"{test_dl.name}: on GPU {self.local_rank} test\t"
                      f"rmse = {self.test_rmses[test_no]}\t"
                      f"mae = {self.test_maes[test_no]}\t"
                      f"corr = {self.test_corrs[test_no]} (p-value={self.test_pvalue[test_no]})")
                with open(self.result_dir + f"/{test_dl.name}_test.res", "a") as t_res:
                    t_res.write(f"{self.epoch+1},{self.test_rmses[test_no]},{self.test_maes[test_no]},{self.test_corrs[test_no]},{self.test_pvalue[test_no]}\n")



