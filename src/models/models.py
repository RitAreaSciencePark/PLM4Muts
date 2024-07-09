# SPDX-FileCopyrightText: 2024 (C) 2024 Marco Celoria <celoria.marco@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import esm
from transformers import  T5EncoderModel
import torch
from torch import nn
import torch.nn.functional as F
import warnings

HIDDEN_UNITS = 768

class ESM2_Finetuning(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="ESM2_Finetuning"
        self.esm_transformer, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        self.fc1 = nn.Linear(2560,HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS,1)
        self.relu=nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def preprocess(self, wild_seq_esm, mut_seq_esm, pos):
        wild_seq_esm = [('', ''.join(wild_seq_esm[0]))]
        mut_seq_esm  = [('', ''.join(mut_seq_esm[0] ))]
        wild_esm_batch_labels, wild_esm_batch_strs, wild_esm_batch_tokens = self.esm_batch_converter(wild_seq_esm)
        mut_esm_batch_labels,  mut_esm_batch_strs,  mut_esm_batch_tokens  = self.esm_batch_converter(mut_seq_esm)
        wild_esm_batch_tokens = wild_esm_batch_tokens
        mut_esm_batch_tokens  =  mut_esm_batch_tokens
        pos = pos
        return wild_esm_batch_tokens, mut_esm_batch_tokens, pos

    def onnx_model_args(self):
        input_names = ["wt", "mut", "pos"]
        output_names = ["ddg"]
        wt  = torch.tensor([[ 0, 15, 15, 15, 14,  4, 13,  6,  9, 19, 18, 11,  4, 16, 12, 10,  6, 10,
                              9, 10, 18,  9, 20, 18, 10,  9,  4, 17,  9,  5,  4,  9,  4, 15, 13,  5,
                             16,  5,  6, 15,  9, 14,  6,  2]])

        mut = torch.tensor([[ 0, 15, 15, 15, 14,  4, 13,  6,  9, 19, 18, 11,  4, 16, 12, 10,  6, 10,
                              9, 10, 18,  9, 20, 18,  5,  9,  4, 17,  9,  5,  4,  9,  4, 15, 13,  5,
                             16,  5,  6, 15,  9, 14,  6,  2]])

        pos  = torch.tensor([23])
        dynamic_axes = {"wt":{ 1: "L"}, "mut":{ 1: "L"}}
        return (wt, mut, pos), (input_names, output_names, dynamic_axes)

    def forward(self, wild_esm_batch_tokens, mut_esm_batch_tokens, pos):
        batch_size   = wild_esm_batch_tokens.shape[0]
        batch_size_m = mut_esm_batch_tokens.shape[0]
        L   = wild_esm_batch_tokens.shape[1]
        L_m = mut_esm_batch_tokens.shape[1]
        assert batch_size == 1
        assert batch_size_m == 1
        assert L_m == L
        # wild_esm_batch_tokens.shape = torch.Size([1, 151]) = torch.Size([batch_size, L]) 
        # wild_esm_batch_tokens.dtype = torch.int64
        # mut_esm_batch_tokens.shape  = torch.Size([1, 151]) = torch.Size([batch_size_m, L_m])
        # mut_esm_batch_tokens.dtype  = torch.int64
        # pos.shape = torch.Size([1])
        # pos.dtype = torch.int64
        wild_esm_reps=self.esm_transformer(wild_esm_batch_tokens,repr_layers=[33])['representations'][33].view(batch_size,L,-1)
        mut_esm_reps=self.esm_transformer(mut_esm_batch_tokens,repr_layers=[33])['representations'][33].view(batch_size,L,-1)
        #wild_esm_reps = wild_esm_reps.reshape(batch_size, L, -1)
        #mut_esm_reps  =  mut_esm_reps.reshape(batch_size, L, -1)
        wild_esm_reps_p = wild_esm_reps[:, pos+1, :].reshape((batch_size,-1))
        mut_esm_reps_p  =  mut_esm_reps[:, pos+1, :].reshape((batch_size,-1))
        wild_esm_reps_m = wild_esm_reps[:, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        mut_esm_reps_m  =  mut_esm_reps[:, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        #esm_reps_p = wild_esm_reps_p - mut_esm_reps_p
        #esm_reps_m = wild_esm_reps_m - mut_esm_reps_m
        outputs = torch.cat((wild_esm_reps_p - mut_esm_reps_p, wild_esm_reps_m - mut_esm_reps_m), dim=1)
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)


class ESM2_Baseline(ESM2_Finetuning):
    def __init__(self):
        ESM2_Finetuning.__init__(self)
        self.name="ESM2_Baseline"
        for param in self.esm_transformer.parameters():
            param.requires_grad = False

class ESM2_Finetuning_OnlyPos(ESM2_Finetuning):
    def __init__(self):
        ESM2_Finetuning.__init__(self)
        self.name="ESM2_Finetuning_OnlyPos"
        self.fc1 = nn.Linear(1280,HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS,1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, wild_esm_batch_tokens, mut_esm_batch_tokens, pos):
        batch_size   = wild_esm_batch_tokens.shape[0]
        batch_size_m = mut_esm_batch_tokens.shape[0]
        L   = wild_esm_batch_tokens.shape[1]
        L_m = mut_esm_batch_tokens.shape[1]
        wild_esm_reps=self.esm_transformer(wild_esm_batch_tokens,repr_layers=[33])['representations'][33].view(batch_size,L,-1)
        mut_esm_reps=self.esm_transformer(mut_esm_batch_tokens,repr_layers=[33])['representations'][33].view(batch_size,L,-1)
        wild_esm_reps_p = wild_esm_reps[:, pos+1, :].reshape((batch_size,-1))
        mut_esm_reps_p  =  mut_esm_reps[:, pos+1, :].reshape((batch_size,-1))
        outputs = wild_esm_reps_p - mut_esm_reps_p
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)


class ESM2_Finetuning_OnlyMean(ESM2_Finetuning):
    def __init__(self):
        ESM2_Finetuning.__init__(self)
        self.name="ESM2_Finetuning_OnlyMean"
        self.fc1 = nn.Linear(1280,HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS,1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, wild_esm_batch_tokens, mut_esm_batch_tokens, pos):
        batch_size   = wild_esm_batch_tokens.shape[0]
        batch_size_m = mut_esm_batch_tokens.shape[0]
        L   = wild_esm_batch_tokens.shape[1]
        L_m = mut_esm_batch_tokens.shape[1]
        wild_esm_reps=self.esm_transformer(wild_esm_batch_tokens,repr_layers=[33])['representations'][33].view(batch_size,L,-1)
        mut_esm_reps=self.esm_transformer(mut_esm_batch_tokens,repr_layers=[33])['representations'][33].view(batch_size,L,-1)
        wild_esm_reps_m = wild_esm_reps[:, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        mut_esm_reps_m  =  mut_esm_reps[:, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        outputs = wild_esm_reps_m - mut_esm_reps_m
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)


class ESM2_Finetuning_Logits(ESM2_Finetuning):
    def __init__(self):
        ESM2_Finetuning.__init__(self)
        self.name="ESM2_Finetuning_Logits"

    def forward(self, wild_esm_batch_tokens, mut_esm_batch_tokens, pos):
        batch_size   = wild_esm_batch_tokens.shape[0]
        batch_size_m = mut_esm_batch_tokens.shape[0]
        L   = wild_esm_batch_tokens.shape[1]
        L_m = mut_esm_batch_tokens.shape[1]
        wild_logits = self.esm_transformer(wild_esm_batch_tokens, repr_layers=[33])['logits']  
        mut_logits = self.esm_transformer(mut_esm_batch_tokens, repr_layers=[33])['logits']
        wild_aminos = "".join([self.esm_alphabet.get_tok(token.item()) for token in wild_esm_batch_tokens[0,:]]).replace('<cls>','')
        mut_aminos = "".join([self.esm_alphabet.get_tok(token.item()) for token in mut_esm_batch_tokens[0,:]]).replace('<cls>','')
        wild_amino_id = self.esm_alphabet.get_idx(wild_aminos[pos])
        mut_amino_id = self.esm_alphabet.get_idx(mut_aminos[pos])
        w_log = wild_logits[:,pos,wild_amino_id].reshape(batch_size,-1)
        m_log = wild_logits[:,pos,mut_amino_id].reshape(batch_size,-1)
        outputs = w_log - m_log
        return outputs


class MSA_Finetuning(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="MSA_Finetuning"
        self.msa_transformer, self.msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_batch_converter = self.msa_alphabet.get_batch_converter()
        self.fc1 = nn.Linear(1536,HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS,1)
        self.relu=nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def onnx_model_args(self):
        input_names = ["wt", "mut", "pos"]
        output_names = ["ddg"]
        wt  = torch.tensor([[[ 0, 10, 14, 13,  5,  5,  6, 21,  7, 17, 12,  5,  9,  5,  7, 16, 16],
                             [18, 15, 11, 15, 12,  5,  9,  7, 11, 11,  8,  4, 15, 16,  9,  5,  9]]])
        
        mut = torch.tensor([[[ 0, 10, 17, 13,  5,  5,  6, 21,  7, 17, 12,  5,  9,  5,  7, 16, 16],
                             [18, 15, 11, 15, 12,  5,  9,  7, 11, 11,  8,  4, 15, 16,  9,  5,  9]]])
        
        pos  = torch.tensor([1])
        dynamic_axes = {"wt":{ 1: "N", 2: "L"}, "mut":{ 1: "N", 2: "L"}}
        return (wt, mut, pos), (input_names, output_names, dynamic_axes) 

    def preprocess(self, wild_seq_msa, mut_seq_msa, pos):
        _, _, wild_msa_batch_tokens = self.msa_batch_converter(wild_seq_msa) 
        _, _, mut_msa_batch_tokens  = self.msa_batch_converter(mut_seq_msa) 
        return wild_msa_batch_tokens, mut_msa_batch_tokens, pos

    def forward(self, wild_msa_batch_tokens, mut_msa_batch_tokens, pos):
        batch_size   = wild_msa_batch_tokens.shape[0]
        batch_size_m = mut_msa_batch_tokens.shape[0]
        N   = wild_msa_batch_tokens.shape[1]
        N_m = mut_msa_batch_tokens.shape[1]
        L   = wild_msa_batch_tokens.shape[2]
        L_m = wild_msa_batch_tokens.shape[2]
        assert batch_size == 1
        assert batch_size_m == 1
        assert N == N_m
        assert L == L_m
        # wild_msa_batch_tokens.shape = torch.Size([1, 98, 163]) = torch.Size([batch_size, N, L])
        # wild_msa_batch_tokens.dtype = torch.int64
        # mut_msa_batch_tokens.shape  = torch.Size([1, 98, 163]) = torch.Size([batch_size_m, N_m, L_m])
        # mut_msa_batch_tokens.dtype  = torch.int64
        # pos.shape = torch.Size([1])
        # pos.dtype = torch.int64
        wild_msa_reps=self.msa_transformer(wild_msa_batch_tokens,repr_layers=[12])['representations'][12].view(batch_size,N,L,-1)
        mut_msa_reps =self.msa_transformer(mut_msa_batch_tokens, repr_layers=[12])['representations'][12].view(batch_size,N,L,-1)
        #wild_msa_reps = wild_msa_reps.reshape(batch_size, N, L, 768)
        #mut_msa_reps  =  mut_msa_reps.reshape(batch_size_m, N_m, L_m, 768)
        wild_msa_reps_p = wild_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        mut_msa_reps_p  =  mut_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        wild_msa_reps_m = wild_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        mut_msa_reps_m  =  mut_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        #msa_reps_p = wild_msa_reps_p - mut_msa_reps_p
        #msa_reps_m = wild_msa_reps_m - mut_msa_reps_m
        outputs = torch.cat((wild_msa_reps_p - mut_msa_reps_p, wild_msa_reps_m - mut_msa_reps_m), dim=1)
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)

class MSA_Baseline(MSA_Finetuning):
    def __init__(self):
        MSA_Finetuning.__init__(self)
        self.name="MSA_Baseline"
        for param in self.msa_transformer.parameters():
            param.requires_grad = False


class MSA_Finetuning_OnlyPos(MSA_Finetuning):
    def __init__(self):
        MSA_Finetuning.__init__(self)
        self.name="MSA_Finetuning_OnlyPos"
        self.fc1 = nn.Linear(768, HIDDEN_UNITS)
        self.fc2 = nn.Linear( HIDDEN_UNITS, 1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, wild_msa_batch_tokens, mut_msa_batch_tokens, pos):
        batch_size   = wild_msa_batch_tokens.shape[0]
        batch_size_m = mut_msa_batch_tokens.shape[0]
        N   = wild_msa_batch_tokens.shape[1]
        N_m = mut_msa_batch_tokens.shape[1]
        L   = wild_msa_batch_tokens.shape[2]
        L_m = wild_msa_batch_tokens.shape[2]
        wild_msa_reps=self.msa_transformer(wild_msa_batch_tokens,repr_layers=[12])['representations'][12].view(batch_size,N,L,-1)
        mut_msa_reps =self.msa_transformer(mut_msa_batch_tokens, repr_layers=[12])['representations'][12].view(batch_size,N,L,-1)
        wild_msa_reps_p = wild_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        mut_msa_reps_p  =  mut_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        outputs = wild_msa_reps_p - mut_msa_reps_p
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)


class MSA_Finetuning_OnlyMean(MSA_Finetuning):
    def __init__(self):
        MSA_Finetuning.__init__(self)
        self.name="MSA_Finetuning_OnlyMean"
        self.fc1 = nn.Linear(768, HIDDEN_UNITS)
        self.fc2 = nn.Linear( HIDDEN_UNITS, 1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, wild_msa_batch_tokens, mut_msa_batch_tokens, pos):
        batch_size   = wild_msa_batch_tokens.shape[0]
        batch_size_m = mut_msa_batch_tokens.shape[0]
        N   = wild_msa_batch_tokens.shape[1]
        N_m = mut_msa_batch_tokens.shape[1]
        L   = wild_msa_batch_tokens.shape[2]
        L_m = wild_msa_batch_tokens.shape[2]
        wild_msa_reps=self.msa_transformer(wild_msa_batch_tokens,repr_layers=[12])['representations'][12].view(batch_size,N,L,-1)
        mut_msa_reps =self.msa_transformer(mut_msa_batch_tokens, repr_layers=[12])['representations'][12].view(batch_size,N,L,-1)
        wild_msa_reps_m = wild_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        mut_msa_reps_m  =  mut_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        outputs = wild_msa_reps_m - mut_msa_reps_m
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)

class MSA_Finetuning_Logits(MSA_Finetuning):
    def __init__(self):
        MSA_Finetuning.__init__(self)
        self.name="MSA_Finetuning_Logits"
   
    def forward(self, wild_msa_batch_tokens, mut_msa_batch_tokens, pos):
        batch_size   = wild_msa_batch_tokens.shape[0]
        batch_size_m = mut_msa_batch_tokens.shape[0]
        N   = wild_msa_batch_tokens.shape[1]
        N_m = mut_msa_batch_tokens.shape[1]
        L   = wild_msa_batch_tokens.shape[2]
        L_m = wild_msa_batch_tokens.shape[2]
        
        wild_logits = self.msa_transformer(wild_msa_batch_tokens, repr_layers=[12])['logits']
        mut_logits = self.msa_transformer(mut_msa_batch_tokens, repr_layers=[12])['logits']
        wild_aminos = "".join([self.msa_alphabet.get_tok(token.item()) for token in wild_msa_batch_tokens[0,0,:]]).replace('<cls>','')
        mut_aminos = "".join([self.msa_alphabet.get_tok(token.item()) for token in mut_msa_batch_tokens[0,0,:]]).replace('<cls>','')
        wild_amino_id = self.msa_alphabet.get_idx(wild_aminos[pos])
        mut_amino_id = self.msa_alphabet.get_idx(mut_aminos[pos])
        w_log = wild_logits[:,0,pos,wild_amino_id].reshape(batch_size,-1)
        m_log = wild_logits[:,0,pos,mut_amino_id].reshape(batch_size,-1)
        outputs = w_log - m_log
     
        return outputs




class ProstT5_Finetuning(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="ProstT5_Finetuning"
        self.prostt5 = T5EncoderModel.from_pretrained("./src/models/models_cache/models--Rostlab--ProstT5/snapshots/d7d097d5bf9a993ab8f68488b4681d6ca70db9e5/", local_files_only=True)
        self.fc1 = nn.Linear(4096,HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS,1)
        self.relu=nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def onnx_model_args(self):
        input_names = ["input_ids", "attention_mask", "pos"]
        output_names = ["ddg"]
        input_ids = torch.tensor([[   [149,  14,   4,  13,  13,   5,  21,   9,  14,   8,  19,   7,   8,   7],
                                      [149,  14,   4,  13,  13,   5,  21,   9,  14,   3,  19,   7,   8,   7],
                                      [148, 135, 128, 135, 138, 141, 139, 135, 146, 135, 128, 135, 138, 131],
                                      [148, 135, 128, 135, 138, 141, 139, 135, 146, 135, 128, 135, 138, 131]
                                 ]])

        attention_mask = torch.tensor([[[1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
                                        [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
                                        [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],
                                        [1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1]
                                      ]])

        pos  = torch.tensor([8])
        dynamic_axes = {"input_ids":{ 1: "N", 2: "L"}, "attention_mask":{ 1: "N", 2: "L"}}
        return (input_ids, attention_mask, pos), (input_names, output_names, dynamic_axes)

    def preprocess(self, input_ids, attention_mask, pos):
        return input_ids, attention_mask, pos

    def forward(self, input_ids, attention_mask, pos):
        #torch.set_printoptions(threshold=10_000)
        batch_size = input_ids.shape[0]
        assert batch_size == 1
        N = input_ids.shape[1]
        L = input_ids.shape[2]
        # input_ids.shape      = torch.Size([1, 4, 231]) = torch.Size([batch_size, N, L])
        # input_ids.dtype = torch.int64
        # attention_mask.shape = torch.Size([1, 4, 231]) = torch.Size([batch_size, N, L])
        # attention_mask.dtype = torch.int64
        # pos.shape = torch.Size([1])
        # pos.dtype = torch.int64
        input_ids = input_ids.reshape((-1, L))
        attention_mask = attention_mask.reshape((-1, L))
        seqs_reps = self.prostt5(input_ids, attention_mask=attention_mask).last_hidden_state
        wild_seq_reps    =    seqs_reps[0].reshape(batch_size, 1, L, -1)
        mut_seq_reps     =    seqs_reps[1].reshape(batch_size, 1, L, -1)
        wild_struct_reps =    seqs_reps[2].reshape(batch_size, 1, L, -1)
        mut_struct_reps  =    seqs_reps[3].reshape(batch_size, 1, L, -1)
        
        wild_seq_reps_p    =      wild_seq_reps[:, :, pos+1, :].reshape((batch_size,-1))
        mut_seq_reps_p     =       mut_seq_reps[:, :, pos+1, :].reshape((batch_size,-1))
        wild_struct_reps_p =   wild_struct_reps[:, :, pos+1, :].reshape((batch_size,-1))
        mut_struct_reps_p  =    mut_struct_reps[:, :, pos+1, :].reshape((batch_size,-1))
        
        wild_seq_reps_m    =    wild_seq_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        mut_seq_reps_m     =     mut_seq_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        wild_struct_reps_m = wild_struct_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        mut_struct_reps_m  =  mut_struct_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        
        seq_reps_p    = wild_seq_reps_p    - mut_seq_reps_p
        struct_reps_p = wild_struct_reps_p - mut_struct_reps_p
        
        seq_reps_m    = wild_seq_reps_m    - mut_seq_reps_m
        struct_reps_m = wild_struct_reps_m - mut_struct_reps_m

        outputs = torch.cat((seq_reps_p, struct_reps_p, seq_reps_m, struct_reps_m), dim=1)
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return outputs


class ProstT5_Baseline(ProstT5_Finetuning):
    def __init__(self):
        ProstT5_Finetuning.__init__(self)
        self.name="ProstT5_Baseline"
        for param in self.prostt5.parameters():
            param.requires_grad = False



