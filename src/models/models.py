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
        self.esm_transformer, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_batch_converter = esm_alphabet.get_batch_converter()
        self.fc1 = nn.Linear(2560,HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS,1)
        self.relu=nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, wild_seq_esm, mut_seq_esm, pos, local_rank):
        wild_seq_esm = [('', ''.join(wild_seq_esm[0]))]
        mut_seq_esm  = [('', ''.join(mut_seq_esm[0] ))]
        wild_esm_batch_labels, wild_esm_batch_strs, wild_esm_batch_tokens = self.esm_batch_converter(wild_seq_esm) 
        mut_esm_batch_labels,  mut_esm_batch_strs,  mut_esm_batch_tokens  = self.esm_batch_converter(mut_seq_esm) 
        batch_size = wild_esm_batch_tokens.shape[0]
        L = wild_esm_batch_tokens.shape[1]
        assert batch_size == 1
        
        wild_esm_batch_tokens = wild_esm_batch_tokens.to(local_rank)
        mut_esm_batch_tokens  =  mut_esm_batch_tokens.to(local_rank)
        pos = pos.to(local_rank)
        wild_esm_reps =self.esm_transformer(wild_esm_batch_tokens, repr_layers=[33])['representations'][33]
        mut_esm_reps  =self.esm_transformer(mut_esm_batch_tokens,  repr_layers=[33])['representations'][33]
        
        wild_esm_reps = wild_esm_reps.reshape(batch_size, L, -1)
        mut_esm_reps  =  mut_esm_reps.reshape(batch_size, L, -1)
        
        wild_esm_reps_p = wild_esm_reps[:, pos+1, :].reshape((batch_size,-1))
        mut_esm_reps_p  =  mut_esm_reps[:, pos+1, :].reshape((batch_size,-1))
        
        wild_esm_reps_m = wild_esm_reps[:, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        mut_esm_reps_m  =  mut_esm_reps[:, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        esm_reps_p = wild_esm_reps_p - mut_esm_reps_p
        esm_reps_m = wild_esm_reps_m - mut_esm_reps_m
        outputs = torch.cat((esm_reps_p, esm_reps_m), dim=1)
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)

class MSA_Baseline(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="MSA_Baseline"
        self.msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_batch_converter = msa_alphabet.get_batch_converter()
        self.fc1 = nn.Linear(1536,HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS,1)
        self.relu=nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        for param in self.msa_transformer.parameters():
            param.requires_grad = False

    def forward(self, wild_seq_msa, mut_seq_msa, pos, local_rank):
        wild_msa_batch_labels, wild_msa_batch_strs, wild_msa_batch_tokens = self.msa_batch_converter(wild_seq_msa) 
        mut_msa_batch_labels,  mut_msa_batch_strs,  mut_msa_batch_tokens  = self.msa_batch_converter(mut_seq_msa) 
        batch_size = wild_msa_batch_tokens.shape[0]
        N = wild_msa_batch_tokens.shape[1]
        M = mut_msa_batch_tokens.shape[1]
        L = wild_msa_batch_tokens.shape[2]
        assert batch_size == 1
        
        wild_msa_batch_tokens = wild_msa_batch_tokens.to(local_rank)
        mut_msa_batch_tokens  =  mut_msa_batch_tokens.to(local_rank)
        pos = pos.to(local_rank)
        wild_msa_reps = self.msa_transformer(wild_msa_batch_tokens, repr_layers=[12])['representations'][12]
        mut_msa_reps  = self.msa_transformer(mut_msa_batch_tokens,  repr_layers=[12])['representations'][12]
        
        wild_msa_reps = wild_msa_reps.reshape(batch_size, N, L, 768)
        mut_msa_reps  =  mut_msa_reps.reshape(batch_size, M, L, 768)
        
        wild_msa_reps_p = wild_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        mut_msa_reps_p  =  mut_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        
        wild_msa_reps_m = wild_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        mut_msa_reps_m  =  mut_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        
        msa_reps_p = wild_msa_reps_p - mut_msa_reps_p
        msa_reps_m = wild_msa_reps_m - mut_msa_reps_m

        outputs = torch.cat((msa_reps_p, msa_reps_m), dim=1)
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)


class MSA_Finetuning(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="MSA_Finetuning"
        self.msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_batch_converter = msa_alphabet.get_batch_converter()
        self.fc1 = nn.Linear(1536,HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS,1)
        self.relu=nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, wild_seq_msa, mut_seq_msa, pos, local_rank):
        wild_msa_batch_labels, wild_msa_batch_strs, wild_msa_batch_tokens = self.msa_batch_converter(wild_seq_msa) 
        mut_msa_batch_labels,  mut_msa_batch_strs,  mut_msa_batch_tokens  = self.msa_batch_converter(mut_seq_msa) 
        batch_size = wild_msa_batch_tokens.shape[0]
        N = wild_msa_batch_tokens.shape[1]
        M = mut_msa_batch_tokens.shape[1]
        L = wild_msa_batch_tokens.shape[2]
        assert batch_size == 1
        
        wild_msa_batch_tokens = wild_msa_batch_tokens.to(local_rank)
        mut_msa_batch_tokens  =  mut_msa_batch_tokens.to(local_rank)
        pos = pos.to(local_rank)
        wild_msa_reps = self.msa_transformer(wild_msa_batch_tokens, repr_layers=[12])['representations'][12]
        mut_msa_reps  = self.msa_transformer(mut_msa_batch_tokens,  repr_layers=[12])['representations'][12]
        
        wild_msa_reps = wild_msa_reps.reshape(batch_size, N, L, 768)
        mut_msa_reps  =  mut_msa_reps.reshape(batch_size, M, L, 768)
        
        wild_msa_reps_p = wild_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        mut_msa_reps_p  =  mut_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        
        wild_msa_reps_m = wild_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        mut_msa_reps_m  =  mut_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        
        msa_reps_p = wild_msa_reps_p - mut_msa_reps_p
        msa_reps_m = wild_msa_reps_m - mut_msa_reps_m

        outputs = torch.cat((msa_reps_p, msa_reps_m), dim=1)
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)

class ProstT5_Finetuning(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="ProstT5_Finetuning"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.fc1 = nn.Linear(4096,HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS,1)
        self.relu=nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, input_ids, attention_mask, pos, local_rank):
        batch_size = input_ids.shape[0]
        assert batch_size == 1
        N = input_ids.shape[1]
        L = input_ids.shape[2]
        pos = pos.to(local_rank)
        input_ids = input_ids.to(local_rank)
        attention_mask = attention_mask.to(local_rank)
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


