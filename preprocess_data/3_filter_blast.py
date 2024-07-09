# SPDX-FileCopyrightText: 2024 (C) 2024 Francesca Cuturello <francesca.cuturello@areasciencepark.it>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import sys
import os

# read files
train = str(sys.argv[1])
test = str(sys.argv[2])
train_db = train.partition('.')[0]
test_db = test.partition('.')[0]

merged_blast = 'blastp/' + test_db +'-' + train_db + '_merged_blast'
values = pd.read_csv(merged_blast, sep=" ")

train_df = pd.read_csv('files/' + test_db +'-'+ train_db +'_checked')
test_df = pd.read_csv('curated_data/' + test)

def blast_results(train_df, test_df, values):
    # calculate alignment overlap fraction over query (test) sequence length
    len_score = []
    identity = []
    evalue = []
    codes = []
    for pdb_train in list(set(train_df['pdb_id'].to_list())):

        pdb_test = [p.partition('-')[0] for p in values['code'].to_list() if p.partition('-')[2] == pdb_train]
        pdb_test = [p for p in pdb_test if p in test_df['pdb_id'].to_list()]
        for pdb in pdb_test:
            len_test = len(test_df[test_df['pdb_id'] == pdb]['wt_seq'].values[0]) 
            overlap = float(values[values['code'] == pdb + '-' + pdb_train]['length'].values[0])
            len_score.append(overlap / len_test)
            identity.append(float(values[values['code'] == pdb + '-' + pdb_train]['identity'].values[0]))
            evalue.append(float(values[values['code'] == pdb + '-' + pdb_train]['evalue'].values[0]))
            codes.append(pdb + '-' + pdb_train)
    
    blast = pd.DataFrame(columns=['code','overlap','identity','evalue'])
    blast['code'] = codes
    blast['overlap'] = len_score
    blast['identity'] = identity
    blast['evalue'] = evalue

    # check for missing pdbs
    all_pairs = list(set([v.partition('-')[2] for v in values['code'].to_list()]))
    for pdb in list(set(train_df['pdb_id'].to_list())):
        if pdb not in all_pairs:
            sys.stdout.write(val+' not found \n')
    
    return blast 

blast = blast_results(train_df, test_df, values)

# filter out homologous sequences from the training set
to_exclude = blast[(((blast['identity'] > 25) & (blast['evalue'] < 0.01) & (blast['overlap'] > 0.5)))]['code'].to_list()
if len(train_df) > 10000:
    to_exclude = to_exclude + blast[blast['evalue']<10**(-3)]['code'].to_list()

wrong_pdbs = list(set([c.partition('-')[2] for c in to_exclude]))
exclud = [x for x in train_df['pdb_id'].to_list() if x in wrong_pdbs]
train_filtered = train_df[~train_df['pdb_id'].isin(exclud)]

# filter training dataset and save excluded
excluded = train_df[train_df['pdb_id'].isin(exclud)]
excluded = pd.DataFrame(list(set(excluded['pdb_id'].to_list())), columns=['pdb'])
excluded.to_csv('files/excluded_'+ train_db + '_for_' + test_db, index=False)

# save resuling dataframe to csv file
csv_name = train_db + '_for_' + test_db
out_dir = 'files/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
train_filtered.to_csv(out_dir + csv_name, index=False)
