import esm
from transformers import  T5EncoderModel
import torch
from torch import nn
import torch.nn.functional as F
import warnings

#warnings.filterwarnings("ignore")

#HIDDEN_UNITS_POS_CONTACT = 5

class MSA_Torino(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="Torino"
        self.msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_batch_converter = msa_alphabet.get_batch_converter()
        self.classifier = nn.Linear(1536,1)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, wild_seq_msa, mut_seq_msa, pos, local_rank):
        wild_msa_batch_labels, wild_msa_batch_strs, wild_msa_batch_tokens = self.msa_batch_converter(wild_seq_msa) 
        mut_msa_batch_labels,  mut_msa_batch_strs,  mut_msa_batch_tokens  = self.msa_batch_converter(mut_seq_msa) 
        batch_size = wild_msa_batch_tokens.shape[0]
        N = wild_msa_batch_tokens.shape[1]
        L = wild_msa_batch_tokens.shape[2]
        assert batch_size == 1
        
        wild_msa_batch_tokens = wild_msa_batch_tokens.to(local_rank)
        mut_msa_batch_tokens  = mut_msa_batch_tokens.to(local_rank)
        pos = pos.to(local_rank)
        wild_msa = self.msa_transformer(wild_msa_batch_tokens, repr_layers=[12])
        mut_msa  = self.msa_transformer(mut_msa_batch_tokens, repr_layers=[12])
        wild_msa_logits = wild_msa['logits']
        wild_msa_reps   = wild_msa['representations'][12]
        mut_msa_logits = mut_msa['logits']
        mut_msa_reps   = mut_msa['representations'][12]
        
        wild_msa_reps = wild_msa_reps.reshape(batch_size, N, L, -1)
        mut_msa_reps  =  mut_msa_reps.reshape(batch_size, N, L, -1)
        
        wild_msa_reps_p = wild_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        mut_msa_reps_p  =  mut_msa_reps[:, 0, pos+1, :].reshape((batch_size,-1))
        
        #wild_msa_reps_m = wild_msa_reps[:, 0, :, :].mean(dim=1).reshape((batch_size,-1))
        #mut_msa_reps_m  =  mut_msa_reps[:, 0, :, :].mean(dim=1).reshape((batch_size,-1))
        
        outputs = torch.cat((wild_msa_reps_p, mut_msa_reps_p), dim=1)

        outputs = self.classifier(outputs)
        return outputs




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

    def forward(self, wild_seq_e, mut_seq_e, struct_e, pos, local_rank):
        wild_seq_e = wild_seq_e.to(local_rank)
        mut_seq_e  =  mut_seq_e.to(local_rank)
        struct_e   =   struct_e.to(local_rank)
        pos = pos.to(local_rank)
        batch_size = wild_seq_e.input_ids.shape[0]
        N = wild_seq_e.input_ids.shape[1]
        L = wild_seq_e.input_ids.shape[2]
        assert batch_size == 1
        
        wild_seq_e.input_ids      = wild_seq_e.input_ids.reshape((-1, L))
        mut_seq_e.input_ids       =  mut_seq_e.input_ids.reshape((-1, L))
        struct_e.input_ids        =   struct_e.input_ids.reshape((-1, L))
        
        wild_seq_e.attention_mask = wild_seq_e.attention_mask.reshape((-1, L))
        mut_seq_e.attention_mask  =  mut_seq_e.attention_mask.reshape((-1, L))
        struct_e.attention_mask   =   struct_e.attention_mask.reshape((-1, L))
        
        wild_seq_reps = self.prostt5(wild_seq_e.input_ids, attention_mask=wild_seq_e.attention_mask).last_hidden_state
        mut_seq_reps  = self.prostt5(mut_seq_e.input_ids,  attention_mask=mut_seq_e.attention_mask).last_hidden_state
        struct_reps   = self.prostt5(struct_e.input_ids,   attention_mask=struct_e.attention_mask).last_hidden_state
        
        wild_seq_reps = wild_seq_reps.reshape(batch_size, N, L, -1)
        mut_seq_reps  = mut_seq_reps.reshape(batch_size, N, L, -1)
        struct_reps   = struct_reps.reshape(batch_size, N, L, -1)
        
        wild_seq_reps_p = wild_seq_reps[:, :, pos+1, :].reshape((batch_size,-1))
        mut_seq_reps_p  =  mut_seq_reps[:, :, pos+1, :].reshape((batch_size,-1))
        struct_reps_p   =   struct_reps[:, :, pos+1, :].reshape((batch_size,-1))
        
        wild_seq_reps_m = wild_seq_reps.mean(dim=2).reshape((batch_size,-1))
        mut_seq_reps_m  = mut_seq_reps.mean(dim=2).reshape((batch_size,-1))
        struct_reps_m   = struct_reps.mean(dim=2).reshape((batch_size,-1))
        
        outputs = torch.cat((wild_seq_reps_p, mut_seq_reps_p, struct_reps_p, 
                             wild_seq_reps_m, mut_seq_reps_m, struct_reps_m), 
                            dim=1)

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
        self.const1 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const2 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const3 = torch.nn.Parameter(torch.ones((1,1024)))

    def forward(self, wild_seq_e, mut_seq_e, struct_e, pos, local_rank):
        wild_seq_e = wild_seq_e.to(local_rank)
        mut_seq_e  =  mut_seq_e.to(local_rank)
        struct_e   =   struct_e.to(local_rank)
        pos = pos.to(local_rank)
        batch_size = wild_seq_e.input_ids.shape[0]
        N = wild_seq_e.input_ids.shape[1]
        L = wild_seq_e.input_ids.shape[2]
        assert batch_size == 1
        wild_seq_e.input_ids = wild_seq_e.input_ids.reshape((-1, L))
        mut_seq_e.input_ids  =  mut_seq_e.input_ids.reshape((-1, L))
        struct_e.input_ids   =   struct_e.input_ids.reshape((-1, L))
        wild_seq_e.attention_mask = wild_seq_e.attention_mask.reshape((-1, L))
        mut_seq_e.attention_mask  =  mut_seq_e.attention_mask.reshape((-1, L))
        struct_e.attention_mask   =   struct_e.attention_mask.reshape((-1, L))
        
        wild_seq_reps = self.prostt5(wild_seq_e.input_ids, attention_mask=wild_seq_e.attention_mask).last_hidden_state
        mut_seq_reps  = self.prostt5(mut_seq_e.input_ids,  attention_mask=mut_seq_e.attention_mask).last_hidden_state
        struct_reps   = self.prostt5(struct_e.input_ids,   attention_mask=struct_e.attention_mask).last_hidden_state
        
        wild_seq_reps = wild_seq_reps.reshape(batch_size, N, L, -1)
        mut_seq_reps  =  mut_seq_reps.reshape(batch_size, N, L, -1)
        struct_reps   =   struct_reps.reshape(batch_size, N, L, -1)
        
        wild_seq_reps_p = wild_seq_reps[:, :, pos+1, :]
        mut_seq_reps_p  =  mut_seq_reps[:, :, pos+1, :]
        struct_reps_p   =   struct_reps[:, :, pos+1, :]
        
        wild_seq_reps_m = wild_seq_reps.mean(dim=2)
        mut_seq_reps_m  =  mut_seq_reps.mean(dim=2)
        struct_reps_m   =   struct_reps.mean(dim=2)
        
        aa_mut_wild_diff_p = mut_seq_reps_p[:, 0, :] - self.const1 * wild_seq_reps_p[:, 0, :] 
        aa_mut_wild_diff_m = mut_seq_reps_m[:, 0, :] - self.const2 * wild_seq_reps_m[:, 0, :] 
        ss_wild_p_m_diff   =  struct_reps_p[:, 0, :] - self.const3 *   struct_reps_m[:, 0, :]

        aa_mut_wild_diff_p = aa_mut_wild_diff_p.reshape((batch_size,-1))
        aa_mut_wild_diff_m = aa_mut_wild_diff_m.reshape((batch_size,-1))
        ss_wild_p_m_diff = ss_wild_p_m_diff.reshape((batch_size,-1))
        
        outputs = torch.cat((aa_mut_wild_diff_p, aa_mut_wild_diff_m, ss_wild_p_m_diff), dim=1)
        
        logits  = self.classifier(outputs)
        return logits

