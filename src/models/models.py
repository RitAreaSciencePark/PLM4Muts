import esm
from transformers import  T5EncoderModel
import torch
from torch import nn
import torch.nn.functional as F
import warnings

#warnings.filterwarnings("ignore")

#HIDDEN_UNITS_POS_CONTACT = 5

class ESM_Torino(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="ESM_Torino"
        self.esm_transformer, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_batch_converter = esm_alphabet.get_batch_converter()
        self.classifier = nn.Linear(2560,1)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(torch.ones((1,1280)))
        self.const2 = torch.nn.Parameter(torch.ones((1,1280)))
        self.const3 = torch.nn.Parameter(torch.ones((1,1280)))
        self.const4 = torch.nn.Parameter(torch.ones((1,1280)))

    def forward(self, wild_seq_esm, mut_seq_esm, pos, local_rank):
        wild_seq_esm = [('' , ''.join(wild_seq_esm[0]))]
        mut_seq_esm = [('' , ''.join(mut_seq_esm[0]))]
        wild_esm_batch_labels, wild_esm_batch_strs, wild_esm_batch_tokens = self.esm_batch_converter(wild_seq_esm) 
        mut_esm_batch_labels,  mut_esm_batch_strs,  mut_esm_batch_tokens  = self.esm_batch_converter(mut_seq_esm) 
        batch_size = wild_esm_batch_tokens.shape[0]
        L = wild_esm_batch_tokens.shape[1]
        assert batch_size == 1
        
        wild_esm_batch_tokens = wild_esm_batch_tokens.to(local_rank)
        mut_esm_batch_tokens  = mut_esm_batch_tokens.to(local_rank)
        pos = pos.to(local_rank)
        wild_esm = self.esm_transformer(wild_esm_batch_tokens, repr_layers=[33])
        mut_esm  = self.esm_transformer(mut_esm_batch_tokens, repr_layers=[33])
        wild_esm_logits = wild_esm['logits']
        wild_esm_reps   = wild_esm['representations'][33]
        mut_esm_logits = mut_esm['logits']
        mut_esm_reps   = mut_esm['representations'][33]
        
        wild_esm_reps = wild_esm_reps.reshape(batch_size, L, -1)
        mut_esm_reps  =  mut_esm_reps.reshape(batch_size, L, -1)
        
        wild_esm_reps_p = wild_esm_reps[:, pos+1, :].reshape((batch_size,-1))
        mut_esm_reps_p  =  mut_esm_reps[:, pos+1, :].reshape((batch_size,-1))
        
        wild_esm_reps_m = wild_esm_reps[:, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        mut_esm_reps_m  =  mut_esm_reps[:, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        esm_reps_p = self.const1 * wild_esm_reps_p - self.const2 *  mut_esm_reps_p
        esm_reps_m = self.const3 * wild_esm_reps_m - self.const4 *  mut_esm_reps_m

        outputs = torch.cat((esm_reps_p, esm_reps_m), dim=1)

        outputs = self.classifier(outputs)
        return outputs


class MSA_Torino(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="MSA_Torino"
        self.msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        self.msa_batch_converter = msa_alphabet.get_batch_converter()
        self.classifier = nn.Linear(1536,1)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(torch.ones((1,768)))
        self.const2 = torch.nn.Parameter(torch.ones((1,768)))
        self.const3 = torch.nn.Parameter(torch.ones((1,768)))
        self.const4 = torch.nn.Parameter(torch.ones((1,768)))

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
        
        wild_msa_reps_m = wild_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        mut_msa_reps_m  =  mut_msa_reps[:, 0, 1:-1, :].mean(dim=1).reshape((batch_size,-1))
        
        msa_reps_p = self.const1 * wild_msa_reps_p - self.const2 *  mut_msa_reps_p
        msa_reps_m = self.const3 * wild_msa_reps_m - self.const4 *  mut_msa_reps_m

        outputs = torch.cat((msa_reps_p, msa_reps_m), dim=1)

        outputs = self.classifier(outputs)
        return outputs




### Model Definition

class ProstT5_Torino(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="ProstT5_Torino"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(2048,1)
        #self.relu=nn.ReLU(inplace=True)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const2 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const3 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const4 = torch.nn.Parameter(torch.ones((1,1024)))
        #self.const5 = torch.nn.Parameter(torch.ones((1,1024)))
        #self.const6 = torch.nn.Parameter(torch.ones((1,1024)))
        #self.const7 = torch.nn.Parameter(torch.ones((1,1024)))
        #self.const8 = torch.nn.Parameter(torch.ones((1,1024)))

    def forward(self, wild_seq_e, mut_seq_e, wild_struct_e, mut_struct_e, pos, local_rank):
        wild_seq_e    = wild_seq_e.to(local_rank)
        mut_seq_e     =  mut_seq_e.to(local_rank)
        wild_struct_e = wild_struct_e.to(local_rank)
        mut_struct_e  = mut_struct_e.to(local_rank)
        pos = pos.to(local_rank)
        batch_size = wild_seq_e.input_ids.shape[0]
        N = wild_seq_e.input_ids.shape[1]
        L = wild_seq_e.input_ids.shape[2]
        assert batch_size == 1
        
        wild_seq_e.input_ids      =    wild_seq_e.input_ids.reshape((-1, L))
        mut_seq_e.input_ids       =     mut_seq_e.input_ids.reshape((-1, L))
        wild_struct_e.input_ids   = wild_struct_e.input_ids.reshape((-1, L))
        mut_struct_e.input_ids    =  mut_struct_e.input_ids.reshape((-1, L))
        
        wild_seq_e.attention_mask    =    wild_seq_e.attention_mask.reshape((-1, L))
        mut_seq_e.attention_mask     =     mut_seq_e.attention_mask.reshape((-1, L))
        wild_struct_e.attention_mask = wild_struct_e.attention_mask.reshape((-1, L))
        mut_struct_e.attention_mask  =  mut_struct_e.attention_mask.reshape((-1, L))
        
        wild_seq_reps   =self.prostt5(wild_seq_e.input_ids,   attention_mask=wild_seq_e.attention_mask).last_hidden_state
        mut_seq_reps    =self.prostt5(mut_seq_e.input_ids,    attention_mask=mut_seq_e.attention_mask).last_hidden_state
        wild_struct_reps=self.prostt5(wild_struct_e.input_ids,attention_mask=wild_struct_e.attention_mask).last_hidden_state
        mut_struct_reps =self.prostt5(mut_struct_e.input_ids, attention_mask=mut_struct_e.attention_mask).last_hidden_state
        
        wild_seq_reps    =    wild_seq_reps.reshape(batch_size, N, L, -1)
        mut_seq_reps     =     mut_seq_reps.reshape(batch_size, N, L, -1)
        wild_struct_reps = wild_struct_reps.reshape(batch_size, N, L, -1)
        mut_struct_reps  =  mut_struct_reps.reshape(batch_size, N, L, -1)
        
        wild_seq_reps_p    =      wild_seq_reps[:, :, pos+1, :].reshape((batch_size,-1))
        mut_seq_reps_p     =       mut_seq_reps[:, :, pos+1, :].reshape((batch_size,-1))
        wild_struct_reps_p =   wild_struct_reps[:, :, pos+1, :].reshape((batch_size,-1))
        mut_struct_reps_p  =    mut_struct_reps[:, :, pos+1, :].reshape((batch_size,-1))
        
        #wild_seq_reps_m    =    wild_seq_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        #mut_seq_reps_m     =     mut_seq_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        #wild_struct_reps_m = wild_struct_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        #mut_struct_reps_m  =  mut_struct_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        
        seq_reps_p    = self.const1 * wild_seq_reps_p    - self.const2 * mut_seq_reps_p
        struct_reps_p = self.const3 * wild_struct_reps_p - self.const4 * mut_struct_reps_p
        
        #seq_reps_m    = self.const5 * wild_seq_reps_m    - self.const6 * mut_seq_reps_m
        #struct_reps_m = self.const7 * wild_struct_reps_m - self.const8 * mut_struct_reps_m

        outputs = torch.cat((seq_reps_p, struct_reps_p), dim=1)
        #outputs = torch.cat((seq_reps_p, struct_reps_p, seq_reps_m, struct_reps_m), dim=1)

        logits = self.classifier(outputs)
        return logits


class ProstT5_Milano(nn.Module):

    def __init__(self):
        super().__init__()
        self.name="ProstT5_Milano"
        self.prostt5 = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
        self.classifier = nn.Linear(4096,1)
        #self.relu=nn.ReLU(inplace=True)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.const1 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const2 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const3 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const4 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const5 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const6 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const7 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const8 = torch.nn.Parameter(torch.ones((1,1024)))

    def forward(self, wild_seq_e, mut_seq_e, wild_struct_e, mut_struct_e, pos, local_rank):
        wild_seq_e    = wild_seq_e.to(local_rank)
        mut_seq_e     =  mut_seq_e.to(local_rank)
        wild_struct_e = wild_struct_e.to(local_rank)
        mut_struct_e  = mut_struct_e.to(local_rank)
        pos = pos.to(local_rank)
        batch_size = wild_seq_e.input_ids.shape[0]
        N = wild_seq_e.input_ids.shape[1]
        L = wild_seq_e.input_ids.shape[2]
        assert batch_size == 1
        
        wild_seq_e.input_ids      =    wild_seq_e.input_ids.reshape((-1, L))
        mut_seq_e.input_ids       =     mut_seq_e.input_ids.reshape((-1, L))
        wild_struct_e.input_ids   = wild_struct_e.input_ids.reshape((-1, L))
        mut_struct_e.input_ids    =  mut_struct_e.input_ids.reshape((-1, L))
        
        wild_seq_e.attention_mask    =    wild_seq_e.attention_mask.reshape((-1, L))
        mut_seq_e.attention_mask     =     mut_seq_e.attention_mask.reshape((-1, L))
        wild_struct_e.attention_mask = wild_struct_e.attention_mask.reshape((-1, L))
        mut_struct_e.attention_mask  =  mut_struct_e.attention_mask.reshape((-1, L))
        
        wild_seq_reps   =self.prostt5(wild_seq_e.input_ids,   attention_mask=wild_seq_e.attention_mask).last_hidden_state
        mut_seq_reps    =self.prostt5(mut_seq_e.input_ids,    attention_mask=mut_seq_e.attention_mask).last_hidden_state
        wild_struct_reps=self.prostt5(wild_struct_e.input_ids,attention_mask=wild_struct_e.attention_mask).last_hidden_state
        mut_struct_reps =self.prostt5(mut_struct_e.input_ids, attention_mask=mut_struct_e.attention_mask).last_hidden_state
        
        wild_seq_reps    =    wild_seq_reps.reshape(batch_size, N, L, -1)
        mut_seq_reps     =     mut_seq_reps.reshape(batch_size, N, L, -1)
        wild_struct_reps = wild_struct_reps.reshape(batch_size, N, L, -1)
        mut_struct_reps  =  mut_struct_reps.reshape(batch_size, N, L, -1)
        
        wild_seq_reps_p    =      wild_seq_reps[:, :, pos+1, :].reshape((batch_size,-1))
        mut_seq_reps_p     =       mut_seq_reps[:, :, pos+1, :].reshape((batch_size,-1))
        wild_struct_reps_p =   wild_struct_reps[:, :, pos+1, :].reshape((batch_size,-1))
        mut_struct_reps_p  =    mut_struct_reps[:, :, pos+1, :].reshape((batch_size,-1))
        
        wild_seq_reps_m    =    wild_seq_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        mut_seq_reps_m     =     mut_seq_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        wild_struct_reps_m = wild_struct_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        mut_struct_reps_m  =  mut_struct_reps[:, :, 1:-1, :].mean(dim=2).reshape((batch_size,-1))
        
        seq_reps_p    = self.const1 * wild_seq_reps_p    - self.const2 * mut_seq_reps_p
        struct_reps_p = self.const3 * wild_struct_reps_p - self.const4 * mut_struct_reps_p
        
        seq_reps_m    = self.const5 * wild_seq_reps_m    - self.const6 * mut_seq_reps_m
        struct_reps_m = self.const7 * wild_struct_reps_m - self.const8 * mut_struct_reps_m

        #outputs = torch.cat((seq_reps_p, struct_reps_p), dim=1)
        outputs = torch.cat((seq_reps_p, struct_reps_p, seq_reps_m, struct_reps_m), dim=1)

        logits = self.classifier(outputs)
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

