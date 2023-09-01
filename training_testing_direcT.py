#import Bio
#from Bio import SeqIO
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy
from scipy import stats
from scipy.stats import pearsonr
from transformers import T5Tokenizer, T5EncoderModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast
import warnings

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

HIDDEN_UNITS_POS_CONTACT = 5



### Model Definition

class ProstT5_Milano(nn.Module):

    def __init__(self):
        super().__init__()
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(4096, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)

    def forward(self, token_ids1, token_ids2, pos):
        outputs1 = self.prostt5.forward(token_ids1).last_hidden_state
        outputs2 = self.prostt5.forward(token_ids2).last_hidden_state
        tmp10=outputs1[:1,pos+1,:]
        tmp11=outputs1[1:,pos+1,:]
        tmp20=outputs2[:1,pos+1,:]
        tmp21=outputs2[1:,pos+1,:]
        # print("a",tmp10.shape) torch.Size([1, 1, 1024])
        outputs=torch.concat((tmp10, tmp11, tmp20, tmp21), dim=2)
        # print("b",outputs.shape) torch.Size([1, 1, 4096])
        logits = self.classifier(outputs)
        # print("c",logits.shape) torch.Size([1, 1, 1])
        return logits

class ProstT5_MilanoMean(nn.Module):

    def __init__(self):
        super().__init__()
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(4096, 1)
        nn.init.xavier_normal_(self.classifier.weight)
        #nn.init.zeros_(self.classifier.bias)

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
        outputs=torch.concat((tmp10, tmp11, tmp20, tmp21), dim=2)
        # print("b",outputs.shape) torch.Size([1, 1, 4096])
        logits = self.classifier(outputs)
        # print("c",logits.shape) torch.Size([1, 1, 1])
        return logits


class ProstT5_Roma(nn.Module):
    def __init__(self):
        super().__init__()
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
        tmp10=outputs1[:1,pos+1,:]
        tmp11=outputs1[1:,pos+1,:]
        tmp20=outputs2[:1,pos+1,:]
        tmp21=outputs2[1:,pos+1,:]
        outputs = torch.concat((tmp10, tmp11, tmp20, tmp21), dim=2)
        outputs = self.relu(self.classifier1(outputs))
        logits  = self.classifier2(outputs)
        return logits

class ProstT5_RomaMean(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(1024, 1)
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
        #print("a",tmp10.shape) torch.Size([1, 1, 1024])
        outputs = self.const1 * tmp10 + self.const2 * tmp20
        #print("b",outputs.shape) torch.Size([1, 1, 1024])
        logits = self.classifier(outputs)
        #print("c", logits.shape) torch.Size([1, 1, 1])
        return logits

class ProstT5_mutLin2Mean(nn.Module):

    def __init__(self):
        super().__init__()
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



### Dataset Definition

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
        ddg = torch.FloatTensor([self.df.iloc[idx]['ddg']])
        return prostt5_batch_tokens1, prostt5_batch_tokens2, pos, torch.unsqueeze(ddg, 0)

    def __len__(self):
        return len(self.df)


### Training and Validation function

def train(epoch):
    scaler = torch.cuda.amp.GradScaler()
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.train()

    for idx, batch in enumerate(training_loader):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a  # free inside reserved
        print(f"idx={idx}/{len(training_loader)}", flush=True) 
        print(f"tot={t};res={r};all={a};free={f}\n", flush=True) 
        input_ids1, input_ids2, pos, labels = batch 
        input_ids1 = input_ids1['input_ids'].to(device)[0] 
        input_ids2 = input_ids2['input_ids'].to(device)[0]
        labels = labels.to(device)
        pos = pos.to(device)
        
        with autocast(dtype=torch.float16):
            
            logits = model(token_ids1 = input_ids1, token_ids2 = input_ids2, pos = pos).to(device)
        #    print("z", logits.shape, labels.shape)
            loss = torch.nn.functional.mse_loss(logits, labels)
        #    loss = torch.nn.functional.l1_loss(logits, labels)
        tr_loss += loss.item()
        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.1)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        tr_labels.extend(labels.cpu().detach())
        tr_preds.extend(logits.cpu().detach())

    epoch_loss = tr_loss / nb_tr_steps

    labels = [id.item() for id in tr_labels]
    predictions = [id.item() for id in tr_preds]

    return labels, predictions


def valid(model, testing_loader):
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels, eval_scores = [], [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            print(f"idx={idx}/{len(testing_loader)}", flush=True) 
            input_ids1, input_ids2, pos, labels = batch
            input_ids1 = input_ids1['input_ids'].to(device)[0]
            input_ids2 = input_ids2['input_ids'].to(device)[0]

            labels = labels.to(device)
            logits = model(token_ids1 = input_ids1, token_ids2 = input_ids2, pos = pos)
            loss = torch.nn.functional.mse_loss(logits, labels)
            #loss = torch.nn.functional.l1_loss(logits, labels)
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
            labels_cpu=labels.cpu().detach()
            logits_cpu=logits.cpu().detach()
            eval_labels.extend(labels_cpu)
            eval_preds.extend(logits_cpu)

    labels = [id.item() for id in eval_labels]
    predictions = [id.item() for id in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    return labels, predictions


# Main

lr = 1e-5 
EPOCHS = 1
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

allmodels = ['ProstT5_Milano', 
             'ProstT5_Roma', 
             'ProstT5_Trieste', 
             'ProstT5_Conconello', 
             'ProstT5_Basovizza',
             'ProstT5_Padriciano',
             'ProstT5_mutLin2', 
             'ProstT5_mutLin4',
             ]

optimizer_name="Adam" # "AdamW"

models = [allmodels[5]]

training_name="fixed_training"
training_name="cut_training"
CurrWorDir = os.getcwd()
print()

#test_path  = CurrWorDir + "/S669_subsets/data/"
#test_files = os.listdir(test_path)

full_df = pd.read_csv('datasets/' + training_name +'_direct.csv',sep=',')
test_datasets = ['datasets/p53_direct.csv','datasets/myoglobin_direct.csv','datasets/ssym_direct.csv','datasets/S669_direct.csv']

#test_datasets = os.listdir(test_path)
#test_datasets = [ "S669_subsets/data/" + f for f in test_datasets]

datasets_path = '/orfeo/scratch/dssc/mceloria/mutations_experiments/ProstT5/'
#for td in test_datasets:
#    print(os.path.join(datasets_path, td))

preds = {n:[] for n in models} 

for model_name in models:
    model_class = globals()[model_name]
    print(f'Training model {model_name}', flush=True) 
    train_df = full_df
    train_ds = ProteinDataset(train_df)
    model = model_class()    
    model.to(device) 
    print("Debug A", flush=True)
    if optimizer_name=="Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    if optimizer_name=="AdamW":
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)

    training_loader = DataLoader(train_ds, batch_size=1, num_workers = 0, shuffle = True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(training_loader), epochs=EPOCHS)
    

    testing_dataframes = [None] * len(test_datasets)
    testing_datasets   = [None] * len(test_datasets)
    testing_loaders    = [None] * len(test_datasets)

    train_loss   = np.zeros(EPOCHS) 
    train_maes   = np.zeros(EPOCHS) 
    val_losses   = [None] * len(test_datasets)
    val_maes     = [None] * len(test_datasets)
    val_corr     = [None] * len(test_datasets)
    print("Debug B", flush=True) 
    for test_no, test_dataset in enumerate(test_datasets):

        testing_dataframes[test_no] = pd.read_csv(os.path.join(datasets_path, test_dataset))
        testing_datasets[test_no]   = ProteinDataset(testing_dataframes[test_no])
        testing_loaders[test_no]    = DataLoader(testing_datasets[test_no], batch_size=1, num_workers = 0)
        val_losses[test_no] = np.zeros(EPOCHS)
        val_maes[test_no]   = np.zeros(EPOCHS)
        val_corr[test_no]   = np.zeros(EPOCHS)
    print("Debug C", flush=True)    
    for epoch in range(EPOCHS):
        labels, predictions =train(epoch)
        L2loss = np.mean((np.array(labels) - np.array(predictions))**2)  
        MAE = np.mean(np.abs(np.array(labels) - np.array(predictions)))
        RMSE=np.sqrt(L2loss)
        train_loss[epoch] = RMSE
        train_maes[epoch] = MAE
        print("-----------------------------------------------------------------------", flush=True) 
        print(f"Training Loss for epoch {epoch+1}/{EPOCHS} - {model_name}: RMSE[{train_loss[epoch]}] - MAE[{train_maes[epoch]}]", flush=True) 
        print("-----------------------------------------------------------------------", flush=True) 

        for test_no, testing_loader in enumerate(testing_loaders):

            labels, predictions = valid(model, testing_loader)
            L2loss = np.mean((np.array(labels) - np.array(predictions))**2) 
            MAE = np.mean(np.abs(np.array(labels) - np.array(predictions)))
            Correlation, p_value=pearsonr(labels, predictions)
            val_maes[test_no][epoch]=MAE
            val_corr[test_no][epoch]=Correlation
            RMSE=np.sqrt(L2loss)
            val_losses[test_no][epoch] = RMSE 
            
            print("***********************************************************************", flush=True) 
            print(f"Validation Dataset - {model_name}:\t{test_datasets[test_no]} len={len(testing_loader)}", flush=True) 
            print(f"Validation RMSE - {model_name}            for epoch {epoch+1}/{EPOCHS}:\t{val_losses[test_no][epoch]}", flush=True) 
            print(f"Validation MAE - {model_name}             for epoch {epoch+1}/{EPOCHS}:\t{val_maes[test_no][epoch]}", flush=True) 
            print(f"Validation Correlation - {model_name}     for epoch {epoch+1}/{EPOCHS}:\t{val_corr[test_no][epoch]}", flush=True) 
            print("***********************************************************************", flush=True) 
           
            if epoch==EPOCHS-1:
                for idx,(lab, pred) in enumerate(zip(labels,predictions)):
                    if np.abs(lab-pred) > 0.0:
                        wild_seq = testing_dataframes[test_no].iloc[idx]['wild_type']
                        mut_seq  = testing_dataframes[test_no].iloc[idx]['mutated']
                        pos = testing_dataframes[test_no].iloc[idx]['pos']
                        ddg = testing_dataframes[test_no].iloc[idx]['ddg']
                        print(f"\n{test_datasets[test_no]}:\nwild_seq={wild_seq}\nmuta_seq={mut_seq}\npos={pos}\nlabels={lab}\tpredictions={pred}\n", 
                              flush=True)
         
    model.to('cpu') 
        
#    torch.save(model.state_dict(), 'weights/' + model_name)
    
    del model
    torch.cuda.empty_cache()

    print("Summary Direct Training", flush=True) 

    test_dataset_names = [ s.rsplit('/', 1)[1].rsplit('.', 1)[0]  for s in test_datasets ] 

    val_loss_names  = [ s + "_RMSE" for s in test_dataset_names]
    val_mae_names   = [ s + "_MAE"  for s in test_dataset_names]
    val_corr_names  = [ s + "_CORR"  for s in test_dataset_names]
    train_loss_dict = {'train_RMSE': train_loss}

    print(val_loss_names)
    print(val_mae_names)
    print(val_corr_names)

    val_loss_df   = pd.DataFrame.from_dict(dict(zip(val_loss_names,  val_losses)))
    val_mae_df    = pd.DataFrame.from_dict(dict(zip(val_mae_names,   val_maes)))
    val_corr_df   = pd.DataFrame.from_dict(dict(zip(val_corr_names,  val_corr)))
    train_loss_df = pd.DataFrame.from_dict(train_loss_dict)

    train_loss_df["epoch"]= train_loss_df.index
    val_corr_df["epoch"]  = val_corr_df.index
    val_mae_df["epoch"]   = val_mae_df.index
    val_loss_df["epoch"]  = val_loss_df.index

    df = pd.concat([frame.set_index("epoch") for frame in [train_loss_df, val_loss_df, val_mae_df, val_corr_df]], axis=1, join="inner").reset_index()

    print(df, flush=True) 

    df.to_csv(f'results3008/Epochs_Statistics_direcT_{model_name}_{training_name}_{optimizer_name}_{lr}_{EPOCHS}_L2.csv')

    plt.figure(figsize=(8,6))
    plt.plot(df["epoch"], df["train_RMSE"],            label="train_direct_RMSE",     color="black",      linestyle="-.")
    plt.plot(df["epoch"], df["p53_direct_RMSE"],       label="p53_direct_RMSE",       color="tab:orange", linestyle="-")
    plt.plot(df["epoch"], df["myoglobin_direct_RMSE"], label="myoglobin_direct_RMSE", color="tab:green",  linestyle="-")
    plt.plot(df["epoch"], df["ssym_direct_RMSE"],      label="ssym_direct_RMSE",      color="tab:red",    linestyle="-")
    plt.plot(df["epoch"], df["S669_direct_RMSE"],      label="S669_direct_RMSE",      color="tab:blue",   linestyle="-")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title(f"Model: {model_name}")
    plt.legend()
    plt.savefig(f'results3008/Epochs_Loss_direcT_{model_name}_{training_name}_{optimizer_name}_{lr}_{EPOCHS}_L2.png')
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.plot(df["epoch"], df["p53_direct_MAE"],       label="p53_direct_MAE",       color="tab:orange")
    plt.plot(df["epoch"], df["myoglobin_direct_MAE"], label="myoglobin_direct_MAE", color="tab:green")
    plt.plot(df["epoch"], df["ssym_direct_MAE"],      label="ssym_direct_MAE",      color="tab:red")
    plt.plot(df["epoch"], df["S669_direct_MAE"],      label="S669_direct_MAE",      color="tab:blue")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.title(f"Model: {model_name}")
    plt.legend()
    plt.savefig(f'results3008/Epochs_MAE_direcT_{model_name}_{training_name}_{optimizer_name}_{lr}_{EPOCHS}_L2.png')
    plt.clf()

    plt.figure(figsize=(8,6))
    plt.plot(df["epoch"], df["p53_direct_CORR"],       label="p53_direct_CORR",       color="tab:orange")
    plt.plot(df["epoch"], df["myoglobin_direct_CORR"], label="myoglobin_direct_CORR", color="tab:green")
    plt.plot(df["epoch"], df["ssym_direct_CORR"],      label="ssym_direct_CORR",      color="tab:red")
    plt.plot(df["epoch"], df["S669_direct_CORR"],      label="S669_direct_CORR",      color="tab:blue")
    plt.xlabel("epoch")
    plt.ylabel("Pearson correlation coefficient")
    plt.title(f"Model: {model_name}")
    plt.legend()
    plt.savefig(f'results3008/Epochs_Pearsonr_direcT_{model_name}_{training_name}_{optimizer_name}_{lr}_{EPOCHS}_L2.png')
    plt.clf()

