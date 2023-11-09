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

    def forward(self, wild_seq_e, mut_seq_e, struct_e, pos):
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
        mut_seq_reps  = mus_seq_reps.reshape(batch_size, N, L, -1)
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

    def forward(self, wild_seq_e, mut_seq_e, struct_e, pos):
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
        self.classifier = nn.Linear(2048,1)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.const1 = torch.nn.Parameter(torch.ones((1,1024)))
        self.const2 = torch.nn.Parameter(torch.ones((1,1024)))

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
        aa_mut_wild_diff_p = self.const1 * reps_p[:, 0, :] - self.const2 * reps_p[:, 1, :] 
        ss_wild_p_m_diff   = reps_p[:, 2, :] 
        aa_mut_wild_diff_p = aa_mut_wild_diff_p.reshape((batch_size,-1))
        ss_wild_p_m_diff   = ss_wild_p_m_diff.reshape((batch_size,-1))
        outputs = torch.cat((aa_mut_wild_diff_p, ss_wild_p_m_diff), dim=1)
        outputs = self.dropout(outputs)
        logits  = self.classifier(outputs)
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




