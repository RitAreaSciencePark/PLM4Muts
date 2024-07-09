import pandas as pd
import numpy as np
import os
import sys

test = sys.argv[2]
train_cap = sys.argv[1]
train_name = train_cap.lower()

train = pd.read_csv('../'+train_cap+'/train/databases/db_' + train_name + '.csv')

if train_name == "s157479":
    original = "mega"
    db_base = "test"
else:
    original = "S3421"
    db_base = "validation"

test_df = pd.read_csv('../'+ train_cap +'/'+ db_base +'/databases/db_'+ test +'.csv')
blast_full = pd.read_csv('../blastp/' + test + '-' + original +'_merged_blast', sep=" ")

pdbs = list(set([m.partition('-')[0] for m in train['code'].to_list()]))
blast_full['train'] = [c.partition('-')[2] for c in blast_full['code'].to_list()]
blast_full['test'] = [c.partition('-')[0] for c in blast_full['code'].to_list()]
blast = blast_full[blast_full['train'].isin(pdbs)].copy()

len_score = []
for i in range(len(blast)):
    pdb = blast.iloc[i]['test']
    len_test = float(len(test_df[test_df['pdb_id'] == pdb]['wt_seq'].values[0]))
    overlap = float(blast.iloc[i]['length'])
    len_score.append(overlap / len_test)
blast['overlap'] = len_score

to_check = blast[(blast['overlap']>0.5)] #&(blast['evalue']<0.5)]

pairs = pd.DataFrame(columns=['test','train'])
pairs['train'] = to_check['train']
pairs['test'] = to_check['test']

if not os.path.exists('parsed'):
    os.makedirs('parsed')
pairs.to_csv(parsed/train_cap+'-'+test+'_pairs', index=False)
