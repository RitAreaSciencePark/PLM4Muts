import pandas as pd
import os
import re
from transformers import T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast
import warnings

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

HIDDEN_UNITS_POS_CONTACT = 5


def from_cvs_files_in_dir_to_dfs_list(dir_path):
    print(dir_path)
    datasets = os.listdir(dir_path)
    print(datasets)
    #datasets_names = [ s.rsplit('/', 1)[1].rsplit('.', 1)[0]  for s in datasets ]
    datasets_names = [ s.rsplit('.', 1)[0]  for s in datasets ]
    print(datasets_names)
    dfs = [None] * len(datasets)
    for i,d in enumerate(datasets):
        d_path = os.path.join(dir_path, d)
        dfs[i] = pd.read_csv(d_path, sep=',')
    return dfs, datasets_names

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

def train(model, training_loader, device, optimizer,scheduler, epoch):
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



def valid(model, testing_loader, device):
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



