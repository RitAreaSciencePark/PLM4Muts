# SPDX-FileCopyrightText: (C) 2024 Francesca Cuturello <francesca.cuturello@areasciencepark.it>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import sys
import os
import shutil
import subprocess

root_dir = os.path.dirname(os.path.realpath('__file__'))
curated_path = os.path.join(root_dir, 'curated_data/')
path = '../datasets/'

db1 = pd.read_csv(curated_path + str(sys.argv[1]))
db2 = pd.read_csv(curated_path + str(sys.argv[2]))
test_df = pd.read_csv(curated_path + str(sys.argv[3]))
val_df = pd.read_csv(curated_path + str(sys.argv[4]))
test = str(sys.argv[3]).partition('.')[0]
val = str(sys.argv[4]).partition('.')[0]

# intersect the two input datasets, cleaned for two different test sets
ref1 = db1['code'].to_list()
ref2 = db2['code'].to_list()

inter = list(set(ref1).intersection(ref2))

if len(ref1) < len(ref2):
    
    merged = db1[db1['code'].isin(inter)]
else:
    merged = db2[db2['code'].isin(inter)]

# create final output dirs structure for training 
train_name = 'S' + str(len(merged))
final_train = path + train_name + '/train/databases/'
if not os.path.exists(final_train):
    os.makedirs(final_train)

final_test = path + train_name + '/test/databases/'
if not os.path.exists(final_test):
    os.makedirs(final_test)

final_val = path + train_name + '/validation/databases/'
if not os.path.exists(final_val):
    os.makedirs(final_val)

to_keep = ['wt_seq','mut_seq','ddg','pdb_id','pos','code','wt_msa']

# add msa path to train
wt_msa_path = merged.apply(lambda x: 'MSA_S' + str(len(merged)) + '/' + x['pdb_id'], axis=1)
merged = merged.assign(wt_msa = wt_msa_path.values)
merged = merged[to_keep]
merged.to_csv(final_train + 'db_s'+ str(len(merged)) + '.csv', index=False)

# add msa path to test
wt_msa_path = test_df.apply(lambda x: 'MSA_'+ test +'/' + x['pdb_id'], axis=1)
test_df = test_df.assign(wt_msa = wt_msa_path.values)
test_df = test_df[to_keep]
test_df.to_csv(final_test + 'db_' + test  + '.csv', index=False)

# add msa path to validation
wt_msa_path = val_df.apply(lambda x: 'MSA_'+ val +'/' + x['pdb_id'], axis=1)
val_df = val_df.assign(wt_msa = wt_msa_path.values)
val_df = val_df[to_keep]
val_df.to_csv(final_val + 'db_'+ val + '.csv', index=False)

# copy msa to final path
shutil.copytree('hhblits_files/' + val, path + train_name + '/validation/MSA_'+ val)

shutil.copytree('hhblits_files/' + test, path + train_name + '/test/MSA_'+ test)

original_train = str(sys.argv[1]).partition('_')[0]
train_msa_path = os.path.join(path + train_name, 'train/MSA_S'+ str(len(merged))+'/')
if not os.path.exists(train_msa_path):
    os.makedirs(train_msa_path)

for file in list(set(merged['pdb_id'].to_list())):
    if not os.path.isfile(train_msa_path + file):
        shutil.copyfile('hhblits_files/' + original_train +'/'+ file, train_msa_path + file)

