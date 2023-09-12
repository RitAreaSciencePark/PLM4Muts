import pandas as pd
import os
import re
import scipy
from scipy import stats
from scipy.stats import pearsonr
from transformers import T5Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast
import warnings

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

HIDDEN_UNITS_POS_CONTACT = 5

def print_peak_memory(prefix, device):
    if device == 0:
	mma = torch.cuda.max_memory_allocated(device)
        mmr = torch.cuda.max_memory_reserved(device)
	tot = torch.cuda.get_device_properties(0).total_memory
        print(f"{prefix}: allocated [{mma // 1e6} MB] - reserved [{mmr // 1e6} MB] - total [{tot // 1e6} MB]")

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

def train_epoch(model, training_loader, device, optimizer,scheduler, epoch):
    scaler = torch.cuda.amp.GradScaler()
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.train()

    for idx, batch in enumerate(training_loader):
	print_peak_memory("Max GPU memory", 0)
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



def valid_epoch(model, testing_loader, device):
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






class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dl: DataLoader,
        val_dls:  list,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        gpu_id: int,
        save_every: int,
        device: torch.cuda.device,
	result_dir: str
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model#.to(gpu_id)
        self.device = "cuda"
        self.train_dl = train_dl
        self.val_dls = val_dls
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
	self.loss_fn = loss_fn
	self.result_dir = result_dir
	
	self.train_logfile = self.result_dir + "/train_metrics.log"
	with open(self.train_logfile, "w") as train_logfile:
    	    train_logfile.write("appended text")

    def train_batch(self, epoch, batch_idx, batch):
        print(f"epoch:{epoch} - batch_idx:{batch_idx}/{len(self.train_dl)}", flush=True)
        print_peak_memory("Max GPU memory", 0)
	input_ids1, input_ids2, pos, labels = batch
        input_ids1 = input_ids1['input_ids'].to(self.device)[0]
        input_ids2 = input_ids2['input_ids'].to(self.device)[0]
        labels = labels.to(self.device)
        pos    =    pos.to(self.device)
        with autocast():
            logits = self.model(token_ids1 = input_ids1, token_ids2 = input_ids2, pos = pos).to(self.device)
	    loss = self.loss_fn(logits, labels) 
        torch.nn.utils.clip_grad_norm_(parameters = self.model.parameters(), max_norm = 0.1)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.scaler.step(self.optimizer)
        self.scheduler.step()
        self.scaler.update()
        self.t_labels.extend(labels.cpu().detach())
        self.t_preds.extend(logits.cpu().detach())

    def train_epoch(self, epoch):
	self.scaler = torch.cuda.amp.GradScaler()
    	self.t_preds, self.t_labels = [], []
    	self.model.train()
	for idx, batch in enumerate(self.train_dl):
		self.train_batch(epoch, idx, batch)
    	t_labels = [id.item() for id in self.t_labels]
    	t_preds  = [id.item() for id in self.t_preds ]
	return t_labels, t_preds

    def save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    
    def valid_batch(self, batch_idx, batch):
        print(f"idx={idx}/{len(testing_loader)}", flush=True)
        input_ids1, input_ids2, pos, labels = batch
        input_ids1 = input_ids1['input_ids'].to(device)[0]
        input_ids2 = input_ids2['input_ids'].to(device)[0]
        labels = labels.to(device)
        logits = self.model(token_ids1 = input_ids1, token_ids2 = input_ids2, pos = pos)
        labels_cpu=labels.cpu().detach()
        logits_cpu=logits.cpu().detach()
        self.eval_labels.extend(labels_cpu)
        self.eval_preds.extend(logits_cpu)

    def valid_epoch(self, epoch, val_dataloader):
    	self.model.eval()
        self.eval_preds, self.eval_labels = [], []
        with torch.no_grad():
            for idx, batch in enumerate(val_dataloader):
		self.valid_batch(idx, batch)
        labels = [id.item() for id in self.eval_labels]
        predictions = [id.item() for id in self.eval_preds]
        return labels, predictions

    def train(self, max_epochs: int):
        self.train_rmse  = torch.zeros(max_epochs)
        self.train_mae   = torch.zeros(max_epochs)
        self.train_corr  = torch.zeros(max_epochs)
        self.val_rmses = [torch.zeros(max_epochs)] * len(self.val_dls)
        self.val_maes  = [torch.zeros(max_epochs)] * len(self.val_dls)
        self.val_corrs = [torch.zeros(max_epochs)] * len(self.val_dls)
        for epoch in range(max_epochs):
            t_labels, t_preds = self.train_epoch(epoch)
            t_mse  = torch.mean(         (torch.tensor(t_labels) - torch.tensor(t_preds))**2)
            t_mae  = torch.mean(torch.abs(torch.tensor(t_labels) - torch.tensor(t_preds))   )
            t_rmse = torch.sqrt(t_mse)
            t_corr, _ = pearsonr(t_labels, t_preds)
            self.train_mae[epoch]  = t_mae
            self.train_corr[epoch] = t_corr
            self.train_rmse[epoch] = t_rmse
            with open("test.txt", "a") as myfile:
               myfile.write(f"{epoch},{self.train_rmse[epoch]},{self.train_mae[epoch]},{self.train_corr[epoch]}")
#            print("***********************************************************************", flush=True)
#            print(f"Training Dataset - {self.model.name}:\tlen={len(self.train_dl)}", flush=True)
#            print(f"Training RMSE - {self.model.name}            for epoch {epoch+1}/{max_epochs}:\t{self.train_rmse[epoch]}", flush=True)
#            print(f"Training MAE - {self.model.name}             for epoch {epoch+1}/{max_epochs}:\t{self.train_mae[epoch]}", flush=True)
#            print(f"Training Correlation - {self.model.name}     for epoch {epoch+1}/{max_epochs}:\t{self.train_corr[epoch]}", flush=True)
#            print("***********************************************************************", flush=True)
#            if epoch % self.save_every == 0:
#                self._save_checkpoint(epoch)
            
            for val_no, val_dl in enumerate(self.val_dls):
                v_labels, v_preds = self.valid_epoch(val_dl)
                v_mse  = torch.mean(         (torch.tensor(v_labels) - torch.tensor(v_preds))**2)
                v_mae  = torch.mean(torch.abs(torch.tensor(v_labels) - torch.tensor(v_preds))   )
                v_rmse = torch.sqrt(v_mse)
                v_corr, _ = pearsonr(v_labels, v_preds)
                self.val_maes[val_no][epoch]  = v_mae
                self.val_corrs[val_no][epoch] = v_corr
                self.val_rmses[val_no][epoch] = v_rmse
                #print("***********************************************************************", flush=True)
                #print(f"Validation Dataset - {self.model.name}:\t{val_names[val_no]} len={len(self.val_dl)}", flush=True)
                #print(f"Validation RMSE - {self.model.name}            for epoch {epoch+1}/{max_epochs}:\t{self.val_rmses[val_no][epoch]}", 
                #      flush=True)
                #print(f"Validation MAE - {self.model.name}             for epoch {epoch+1}/{max_epochs}:\t{self.val_maes[val_no][epoch]}", 
                #      flush=True)
                #print(f"Validation Correlation - {self.model.name}     for epoch {epoch+1}/{max_epochs}:\t{self.val_corrs[val_no][epoch]}", 
                #      flush=True)
                #print("***********************************************************************", flush=True)

                if epoch==max_epochs - 1:
                    for idx,(lab, pred) in enumerate(zip(v_labels,v_preds)):
                        if np.abs(lab-pred) > 0.0:
                            wild_seq = val_dfs[val_no].iloc[idx]['wild_type']
                            mut_seq  = val_dfs[val_no].iloc[idx]['mutated']
                            pos = val_dfs[val_no].iloc[idx]['pos']
                            ddg = val_dfs[val_no].iloc[idx]['ddg']
                            print(f"{val_names[val_no]}:\nwild_seq={wild_seq}\nmuta_seq={mut_seq}\npos={pos}\nlabels={lab}\tpredictions={pred}\n",
                                  flush=True)

        model.to('cpu')

        #    torch.save(model.state_dict(), 'weights/' + self.model.name)

        del model
        torch.cuda.empty_cache()

        print("Summary Direct Training", flush=True)
        val_rmse_names  = [ s + "_rmse" for s in val_names]
        val_mae_names   = [ s + "_mae"  for s in val_names]
        val_corr_names  = [ s + "_corr" for s in val_names]
        train_rmse_dict = {'train_rmse': train_rmse}
        train_mae_dict  = {'train_mae':  train_mae}
        train_corr_dict = {'train_corr': train_corr}

        val_rmse_df   = pd.DataFrame.from_dict(dict(zip(val_rmse_names,  val_rmses)))
        val_mae_df    = pd.DataFrame.from_dict(dict(zip(val_mae_names,   val_maes)))
        val_corr_df   = pd.DataFrame.from_dict(dict(zip(val_corr_names,  val_corrs)))
        train_rmse_df = pd.DataFrame.from_dict(train_rmse_dict)
        train_mae_df  = pd.DataFrame.from_dict(train_mae_dict)
        train_corr_df = pd.DataFrame.from_dict(train_corr_dict)

        train_rmse_df["epoch"] = train_rmse_df.index
        train_mae_df["epoch"]  = train_mae_df.index
        train_corr_df["epoch"] = train_corr_df.index
        val_corr_df["epoch"]   = val_corr_df.index
        val_mae_df["epoch"]    = val_mae_df.index
        val_rmse_df["epoch"]   = val_rmse_df.index

        df = pd.concat([frame.set_index("epoch") for frame in [train_rmse_df, train_mae_df, train_corr_df, val_rmse_df, val_mae_df, val_corr_df]],
               axis=1, join="inner").reset_index()

        print(df, flush=True)

        if not(os.path.exists(result_dir) and os.path.isdir(result_dir)):
            os.makedirs(result_dir)

        df.to_csv( result_dir + "/epochs_statistics.csv")
        colors = cm.rainbow(torch.linspace(0, 1, len(val_rmse_names)))
        plt.figure(figsize=(8,6))
        plt.plot(df["epoch"], df["train_rmse"],      label='train_rmse',  color="black",   linestyle="-.")
        for i, val_rmse_name in enumerate(val_rmse_names):
            plt.plot(df["epoch"], df[val_rmse_name], label=val_rmse_name, color=colors[i], linestyle="-")
        plt.xlabel("epoch")
        plt.ylabel("RMSE")
        plt.title(f"Model: {self.model.name}")
        plt.legend()
        plt.savefig( result_dir + "/epochs_rmse.png")
        plt.clf()

        plt.figure(figsize=(8,6))
        plt.plot(df["epoch"], df["train_mae"],      label='train_mae',  color="black",   linestyle="-.")
        for i, val_mae_name in enumerate(val_mae_names):
            plt.plot(df["epoch"], df[val_mae_name], label=val_mae_name, color=colors[i], linestyle="-")
        plt.xlabel("epoch")
        plt.ylabel("MAE")
        plt.title(f"Model: {self.model.name}")
        plt.legend()
        plt.savefig( result_dir + "/epochs_mae.png")
        plt.clf()

        plt.figure(figsize=(8,6))
        plt.plot(df["epoch"], df["train_corr"],      label='train_corr',  color="black",   linestyle="-.")
        for i, val_corr_name in enumerate(val_corr_names):
            plt.plot(df["epoch"], df[val_corr_name], label=val_corr_name, color=colors[i], linestyle="-")
        plt.xlabel("epoch")
        plt.ylabel("Pearson correlation coefficient")
        plt.title(f"Model: {self.model.name}")
        plt.legend()
        plt.savefig(result_dir + "/epochs_pearsonr.png")
        plt.clf()

