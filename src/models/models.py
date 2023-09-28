#import math
#import matplotlib.pyplot as plt
#import numpy as np
#import os
#import pandas as pd
#import re
#import scipy
#from scipy import stats
#from scipy.stats import pearsonr
from transformers import  T5EncoderModel
import torch
from torch import nn
#from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
#from torch.cuda.amp import autocast
import warnings

#torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

#HIDDEN_UNITS_POS_CONTACT = 5



### Model Definition

class ProstT5_Milano(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="Milano"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier1 = nn.Linear(6144,1024)
        self.classifier2 = nn.Linear(1024,1)
        self.relu=nn.ReLU(inplace=True)
        nn.init.xavier_normal_(self.classifier1.weight)
        nn.init.xavier_normal_(self.classifier2.weight)
        nn.init.zeros_(self.classifier1.bias)
        nn.init.zeros_(self.classifier2.bias)

    def forward(self, seqs, pos):
        batch_size=seqs.input_ids.shape[0]
        N = seqs.input_ids.shape[1]
        L = seqs.input_ids.shape[2]
        assert batch_size == 1
        seqs.input_ids      = seqs.input_ids.reshape((-1, L))
        seqs.attention_mask = seqs.attention_mask.reshape((-1, L))
        reps = self.prostt5(seqs.input_ids,attention_mask=seqs.attention_mask).last_hidden_state
        reps = reps.reshape(batch_size, N, L, -1)
        reps_p = reps[:, :, pos+1, :]
        reps_m = reps.mean(dim=2)
        reps_p = reps_p.reshape((batch_size,-1))
        reps_m = reps_m.reshape((batch_size,-1))
        outputs = torch.cat((reps_p, reps_m), dim=1)
        outputs = self.relu(self.classifier1(outputs))
        logits  = self.classifier2(outputs)
        return logits


class ProstT5_Roma(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="Roma"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(3072,1)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, seqs, pos):
        batch_size=seqs.input_ids.shape[0]
        N = seqs.input_ids.shape[1]
        L = seqs.input_ids.shape[2]
        assert batch_size == 1
        seqs.input_ids      = seqs.input_ids.reshape((-1, L))
        seqs.attention_mask = seqs.attention_mask.reshape((-1, L))
        reps = self.prostt5(seqs.input_ids,attention_mask=seqs.attention_mask).last_hidden_state
        reps = reps.reshape(batch_size, N, L, -1)
        reps_p = reps[:, :, pos+1, :]
        reps_m = reps.mean(dim=2)
        aa_mut_wild_diff_p = reps_p[:, 0, :] - reps_p[:, 1, :] 
        aa_mut_wild_diff_m = reps_m[:, 0, :] - reps_m[:, 1, :] 
        ss_wild_p_m_diff   = reps_p[:, 2, :] - reps_m[:, 2, :]
        aa_mut_wild_diff_p = aa_mut_wild_diff_p.reshape((batch_size,-1))
        aa_mut_wild_diff_m = aa_mut_wild_diff_m.reshape((batch_size,-1))
        ss_wild_p_m_diff = ss_wild_p_m_diff.reshape((batch_size,-1))
        outputs = torch.cat((aa_mut_wild_diff_p, aa_mut_wild_diff_m, ss_wild_p_m_diff), dim=1)
        logits  = self.classifier(outputs)
        return logits

class ProstT5_RomaMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="RomaMean"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier1 = nn.Linear(4096, 1024)
        nn.init.xavier_normal_(self.classifier1.weight)
        #nn.init.zeros_(self.classifier1.bias)
        self.classifier2 = nn.Linear(1024, 1)
        nn.init.xavier_normal_(self.classifier2.weight)
        #nn.init.zeros_(self.classifier2.bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,1:-1,:]
        tmp11=outputs1[1:,1:-1,:]
        tmp20=outputs2[:1,1:-1,:]
        tmp21=outputs2[1:,1:-1,:]
        tmp10 = tmp10.mean(dim=1, keepdim=True)
        tmp11 = tmp11.mean(dim=1, keepdim=True)
        tmp20 = tmp20.mean(dim=1, keepdim=True)
        tmp21 = tmp21.mean(dim=1, keepdim=True)
        outputs = torch.concat((tmp10, tmp11, tmp20, tmp21), dim=2)
        outputs = self.relu(self.classifier1(outputs))
        logits  = self.classifier2(outputs)
        return logits


class ProstT5_Trieste(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="Trieste"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier1 = nn.Linear(2048, 1024)
        self.classifier2 = nn.Linear(2048, 1024)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))
        self.classifier3 = nn.Linear(1024, 1)
        nn.init.xavier_normal_(self.classifier1.weight)
        nn.init.xavier_normal_(self.classifier2.weight)
        nn.init.xavier_normal_(self.classifier3.weight)
        #nn.init.zeros_(self.classifier1.bias)
        #nn.init.zeros_(self.classifier2.bias)
        #nn.init.zeros_(self.classifier3.bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,pos+1,:]
        tmp11=outputs1[1:,pos+1,:]
        tmp20=outputs2[:1,pos+1,:]
        tmp21=outputs2[1:,pos+1,:]
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        tmp1=torch.concat((tmp10, tmp11), dim=2)
        tmp2=torch.concat((tmp20, tmp21), dim=2)
        # print("b",tmp1.shape) torch.Size([1, 1, 2048])
        outputs1 = self.relu1(self.classifier1(tmp1))
        outputs2 = self.relu2(self.classifier2(tmp2))
        # print("c",outputs1.shape) torch.Size([1, 1, 1024])
        outputs = self.const1 * outputs1 + self.const2 * outputs2
        # print("d",outputs.shape) torch.Size([1, 1, 1024])
        logits = self.classifier3(outputs)
        # print("e",logits.shape) torch.Size([1, 1, 1])
        return logits


class ProstT5_TriesteMean(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="TriesteMean"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier1 = nn.Linear(2048, 1024)
        self.classifier2 = nn.Linear(2048, 1024)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))
        self.classifier3 = nn.Linear(1024, 1)
        nn.init.xavier_normal_(self.classifier1.weight)
        nn.init.xavier_normal_(self.classifier2.weight)
        nn.init.xavier_normal_(self.classifier3.weight)
        #nn.init.zeros_(self.classifier1.bias)
        #nn.init.zeros_(self.classifier2.bias)
        #nn.init.zeros_(self.classifier3.bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,1:-1,:]
        tmp11=outputs1[1:,1:-1,:]
        tmp20=outputs2[:1,1:-1,:]
        tmp21=outputs2[1:,1:-1,:]
        tmp10 = tmp10.mean(dim=1, keepdim=True)
        tmp11 = tmp11.mean(dim=1, keepdim=True)
        tmp20 = tmp20.mean(dim=1, keepdim=True)
        tmp21 = tmp21.mean(dim=1, keepdim=True)
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        tmp1=torch.concat((tmp10, tmp11), dim=2)
        tmp2=torch.concat((tmp20, tmp21), dim=2)
        # print("b",tmp1.shape) torch.Size([1, 1, 2048])
        outputs1 = self.relu1(self.classifier1(tmp1))
        outputs2 = self.relu2(self.classifier2(tmp2))
        # print("c",outputs1.shape) torch.Size([1, 1, 1024])
        outputs = self.const1 * outputs1 + self.const2 * outputs2
        # print("d",outputs.shape) torch.Size([1, 1, 1024])
        logits = self.classifier3(outputs)
        # print("e",logits.shape) torch.Size([1, 1, 1])
        return logits


class ProstT5_Conconello(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="Conconello"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier1 = nn.Linear(2048, 1024)
        self.classifier2 = nn.Linear(2048, 1024)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))
        self.classifier3 = nn.Linear(1024, 1)
        nn.init.xavier_normal_(self.classifier1.weight)
        nn.init.xavier_normal_(self.classifier2.weight)
        nn.init.xavier_normal_(self.classifier3.weight)
        #nn.init.zeros_(self.classifier1.bias)
        #nn.init.zeros_(self.classifier2.bias)
        #nn.init.zeros_(self.classifier3.bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,pos+1,:]
        tmp11=outputs1[1:,pos+1,:]
        tmp20=outputs2[:1,pos+1,:]
        tmp21=outputs2[1:,pos+1,:]
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        tmp1=torch.concat((tmp10, tmp20), dim=2)
        tmp2=torch.concat((tmp11, tmp21), dim=2)
        # print("b",tmp1.shape) torch.Size([1, 1, 2048])
        outputs1 = self.relu1(self.classifier1(tmp1))
        outputs2 = self.relu2(self.classifier2(tmp2))
        # print("c",outputs1.shape) torch.Size([1, 1, 1024])
        outputs = self.const1 * outputs1 + self.const2 * outputs2
        # print("d",outputs.shape) torch.Size([1, 1, 1024])
        logits = self.classifier3(outputs)
        # print("e",logits.shape) torch.Size([1, 1, 1])
        return logits

class ProstT5_ConconelloMean(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="ConconelloMean"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier1 = nn.Linear(2048, 1024)
        self.classifier2 = nn.Linear(2048, 1024)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))
        self.classifier3 = nn.Linear(1024, 1)
        nn.init.xavier_normal_(self.classifier1.weight)
        nn.init.xavier_normal_(self.classifier2.weight)
        nn.init.xavier_normal_(self.classifier3.weight)
        #nn.init.zeros_(self.classifier1.bias)
        #nn.init.zeros_(self.classifier2.bias)
        #nn.init.zeros_(self.classifier3.bias)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,1:-1,:]
        tmp11=outputs1[1:,1:-1,:]
        tmp20=outputs2[:1,1:-1,:]
        tmp21=outputs2[1:,1:-1,:]
        tmp10 = tmp10.mean(dim=1, keepdim=True)
        tmp11 = tmp11.mean(dim=1, keepdim=True)
        tmp20 = tmp20.mean(dim=1, keepdim=True)
        tmp21 = tmp21.mean(dim=1, keepdim=True)
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        tmp1=torch.concat((tmp10, tmp20), dim=2)
        tmp2=torch.concat((tmp11, tmp21), dim=2)
        # print("b",tmp1.shape) torch.Size([1, 1, 2048])
        outputs1 = self.relu1(self.classifier1(tmp1))
        outputs2 = self.relu2(self.classifier2(tmp2))
        # print("c",outputs1.shape) torch.Size([1, 1, 1024])
        outputs = self.const1 * outputs1 + self.const2 * outputs2
        # print("d",outputs.shape) torch.Size([1, 1, 1024])
        logits = self.classifier3(outputs)
        # print("e",logits.shape) torch.Size([1, 1, 1])
        return logits


class ProstT5_Basovizza(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="Basovizza"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(2048, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))
        self.const3 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const4 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,pos+1,:]
        tmp11=outputs1[1:,pos+1,:]
        tmp20=outputs2[:1,pos+1,:]
        tmp21=outputs2[1:,pos+1,:]
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        linear_seq    = self.const1 * tmp10 + self.const2 * tmp20
        linear_struct = self.const3 * tmp11 + self.const4 * tmp21
        # print("b",linear_seq.shape) torch.Size([1, 1, 1024])
        outputs = torch.concat((linear_seq, linear_struct), dim=2)
        # print("c",outputs.shape) torch.Size([1, 1, 2048])
        logits  = self.classifier(outputs)
        # print("d", logits.shape) torch.Size([1, 1, 1])
        return logits

class ProstT5_BasovizzaMean(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="BasovizzaMean"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(2048, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))
        self.const3 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const4 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,1:-1,:]
        tmp11=outputs1[1:,1:-1,:]
        tmp20=outputs2[:1,1:-1,:]
        tmp21=outputs2[1:,1:-1,:]
        tmp10 = tmp10.mean(dim=1, keepdim=True)
        tmp11 = tmp11.mean(dim=1, keepdim=True)
        tmp20 = tmp20.mean(dim=1, keepdim=True)
        tmp21 = tmp21.mean(dim=1, keepdim=True)
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        linear_seq    = self.const1 * tmp10 + self.const2 * tmp20
        linear_struct = self.const3 * tmp11 + self.const4 * tmp21
        # print("b",linear_seq.shape) torch.Size([1, 1, 1024])
        outputs = torch.concat((linear_seq, linear_struct), dim=2)
        # print("c",outputs.shape) torch.Size([1, 1, 2048])
        logits  = self.classifier(outputs)
        # print("d", logits.shape) torch.Size([1, 1, 1])
        return logits


class ProstT5_Padriciano(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="Padriciano"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(2048, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,pos+1,:]
        tmp11=outputs1[1:,pos+1,:]
        tmp20=outputs2[:1,pos+1,:]
        tmp21=outputs2[1:,pos+1,:]
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        linear_seq = self.const1 * tmp10 + self.const2 * tmp20
        # print("b",linear_seq.shape) torch.Size([1, 1, 1024])
        outputs = torch.concat((linear_seq, tmp11), dim=2)
        # print("c",outputs.shape) torch.Size([1, 1, 2048])
        logits  = self.classifier(outputs)
        # print("d", logits.shape) torch.Size([1, 1, 1])
        return logits

class ProstT5_PadricianoMean(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="PadricianoMean"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(2048, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,1:-1,:]
        tmp11=outputs1[1:,1:-1,:]
        tmp20=outputs2[:1,1:-1,:]
        tmp21=outputs2[1:,1:-1,:]
        tmp10 = tmp10.mean(dim=1, keepdim=True)
        tmp11 = tmp11.mean(dim=1, keepdim=True)
        tmp20 = tmp20.mean(dim=1, keepdim=True)
        tmp21 = tmp21.mean(dim=1, keepdim=True)
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        linear_seq = self.const1 * tmp10 + self.const2 * tmp20
        # print("b",linear_seq.shape) torch.Size([1, 1, 1024])
        outputs = torch.concat((linear_seq, tmp11), dim=2)
        # print("c",outputs.shape) torch.Size([1, 1, 2048])
        logits  = self.classifier(outputs)
        # print("d", logits.shape) torch.Size([1, 1, 1])
        return logits



class ProstT5_mutLin2(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="mutLin2"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(1024, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        print("o", outputs1.shape, pos.shape)
        tmp10=outputs1[:1,pos+1,:]
        tmp11=outputs1[1:,pos+1,:]
        tmp20=outputs2[:1,pos+1,:]
        tmp21=outputs2[1:,pos+1,:]
        print("a",tmp10.shape) #torch.Size([1, 1, 1024])
        outputs = self.const1 * tmp10 + self.const2 * tmp20
        print("b",outputs.shape) #torch.Size([1, 1, 1024])
        logits = self.classifier(outputs)
        print("c", logits.shape)# torch.Size([1, 1, 1])
        return logits

class ProstT5_mutLin2Mean(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="mutLin2Mean"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(1024, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(      torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter( -1 * torch.ones((1,1,1024)))

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,1:-1,:]
        tmp11=outputs1[1:,1:-1,:]
        tmp20=outputs2[:1,1:-1,:]
        tmp21=outputs2[1:,1:-1,:]
        tmp10 = tmp10.mean(dim=1, keepdim=True)
        tmp11 = tmp11.mean(dim=1, keepdim=True)
        tmp20 = tmp20.mean(dim=1, keepdim=True)
        tmp21 = tmp21.mean(dim=1, keepdim=True)
        #print("a",tmp10.shape) torch.Size([1, 1, 1024])
        outputs = self.const1 * tmp10 + self.const2 * tmp20
        #print("b",outputs.shape) torch.Size([1, 1, 1024])
        logits = self.classifier(outputs)
        #print("c", logits.shape) torch.Size([1, 1, 1])
        return logits




class ProstT5_mutLin4(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="mutLin4"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(1024, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(     torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter(-1 * torch.ones((1,1,1024)))
        self.const3 = torch.nn.Parameter(     torch.ones((1,1,1024)))
        self.const4 = torch.nn.Parameter(-1 * torch.ones((1,1,1024)))

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,pos+1,:]
        tmp11=outputs1[1:,pos+1,:]
        tmp20=outputs2[:1,pos+1,:]
        tmp21=outputs2[1:,pos+1,:]
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        outputs = self.const1 * tmp10 + self.const2 * tmp20 + self.const3 * tmp11 + self.const4 * tmp21
        # print("b",outputs.shape) torch.Size([1, 1, 1024])
        logits = self.classifier(outputs)
        # print("c", logits.shape) torch.Size([1, 1, 1])
        return logits


class ProstT5_mutLin4Mean(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="mutLin4Mean"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(1024, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(     torch.ones((1,1,1024)))
        self.const2 = torch.nn.Parameter(-1 * torch.ones((1,1,1024)))
        self.const3 = torch.nn.Parameter(     torch.ones((1,1,1024)))
        self.const4 = torch.nn.Parameter(-1 * torch.ones((1,1,1024)))

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,1:-1,:]
        tmp11=outputs1[1:,1:-1,:]
        tmp20=outputs2[:1,1:-1,:]
        tmp21=outputs2[1:,1:-1,:]
        tmp10 = tmp10.mean(dim=1, keepdim=True)
        tmp11 = tmp11.mean(dim=1, keepdim=True)
        tmp20 = tmp20.mean(dim=1, keepdim=True)
        tmp21 = tmp21.mean(dim=1, keepdim=True)
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        outputs = self.const1 * tmp10 + self.const2 * tmp20 + self.const3 * tmp11 + self.const4 * tmp21
        # print("b",outputs.shape) torch.Size([1, 1, 1024])
        logits = self.classifier(outputs)
        # print("c", logits.shape) torch.Size([1, 1, 1])
        return logits




#class ProstT5_mut(nn.Module):
#
#    def __init__(self):
#        super().__init__()
#        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
#        self.classifier = nn.Linear(2048, 1)
#        self.const1 = torch.nn.Parameter(torch.ones((1,2048)))
#        self.const2 = torch.nn.Parameter(-1 * torch.ones((1,2048)))

#    def forward(self, token_ids1, token_ids2, pos):
#        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
#        tmp1 = outputs1[:,pos+1,:]
#        merged_out1 = tmp1.reshape(1,-1)
#        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
#        tmp2 = outputs2[:,pos+1,:]
#        merged_out2 = tmp2.reshape(1,-1)
#        outputs = self.const1 * merged_out1 + self.const2 * merged_out2
#        logits = self.classifier(outputs)
#        logits = logits.unsqueeze(0)
#        return logits

#class ProstT5_mutMLP(nn.Module):

#    def __init__(self):
#        super().__init__()
#        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
#        self.classifier1 = nn.Linear(2048, 512)
#        self.classifier2 = nn.Linear(512, 1)
#        self.const1 = torch.nn.Parameter(torch.ones((1,2048)))
#        self.const2 = torch.nn.Parameter(-1 * torch.ones((1,2048)))
#        self.relu = nn.ReLU(inplace=True)

#    def forward(self, token_ids1, token_ids2, pos):
#        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
#        tmp1 = outputs1[:,pos+1,:]
#        merged_out1 = tmp1.reshape(1,-1)
#        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
#        tmp2 = outputs2[:,pos+1,:]
#        merged_out2 = tmp2.reshape(1,-1)
#        outputs = self.const1 * merged_out1 + self.const2 * merged_out2
#        logits = self.relu(self.classifier1(outputs))
#        logits = self.classifier2(logits)
#        logits = logits.unsqueeze(0)
#        return logits

#class ProstT5_mutSeqs(nn.Module):
#
#    def __init__(self):
#        super().__init__()
#        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
#        self.classifier = nn.Linear(1024, 1)
#        self.const1 = torch.nn.Parameter(torch.ones((1,1024)))
#        self.const2 = torch.nn.Parameter(-1 * torch.ones((1,1024)))
#
#    def forward(self, token_ids1, token_ids2, pos):
#        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
#        tmp1 = outputs1[:1,pos+1,:]
#        merged_out1 = tmp1.reshape(1,-1)
#        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
#        tmp2 = outputs2[:1,pos+1,:]
#        merged_out2 = tmp2.reshape(1,-1)
#        outputs = self.const1 * merged_out1 + self.const2 * merged_out2
#        logits = self.classifier(outputs)
#        logits = logits.unsqueeze(0)
#        return logits



#class ProstT5_mutLin4(nn.Module):
#
#    def __init__(self):
#        super().__init__()
#        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
#        self.classifier = nn.Linear(1024, 1)
#        self.const1 = torch.nn.Parameter(torch.ones((1,1024)))
#        self.const2 = torch.nn.Parameter(-1 * torch.ones((1,1024)))
#        self.const3 = torch.nn.Parameter(torch.ones((1,1024)))
#        self.const4 = torch.nn.Parameter(-1 * torch.ones((1,1024)))
#
#    def forward(self, token_ids1, token_ids2, pos):
#        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
#        tmp1 = outputs1[:,pos+1,:]
#        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
#        tmp2 = outputs2[:,pos+1,:]
#        outputs = self.const1 * tmp1[:1,:] + self.const2 * tmp2[:1,:] + self.const3 * tmp1[1:,:] + self.const4 * tmp2[1:,:]
#        logits = self.classifier(outputs)
#        logits = logits.unsqueeze(0)
#        return logits




