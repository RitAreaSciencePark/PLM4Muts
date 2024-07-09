# SPDX-FileCopyrightText: (C) 2024 Francesca Cuturello <francesca.cuturello@areasciencepark.it>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import numpy as np
import os
import sys


def check_columns(hard_coded_cols, df):

# The initial dataframes must contain 'wt_seq', 'mut_seq', 'ddg', 'pdb_id', 'pos' and either 'mut_info' or 'code'
# define required columns

    if 'code' in df.columns:
        columns =  hard_coded_cols + ['code'] 
    else:
        columns =  hard_coded_cols + ['mut_info']
    
    return columns



def parse_columns(df):
    
# add 'code' column if not present

    if 'code' not in df.columns:
        pdbs = [p + '-' for p in df['pdb_id'].to_list() ]
        codes = [p+str(n) for p,n in zip(pdbs,df['mut_info'].to_list())]
        df['code'] = codes

    return df


def non_common_prots(train_df, test_df):

#   based on pdb names, exclude from training proteins named as in test set
    
    train_pdbs = list(set(train_df['pdb_id'].to_list()))
    test_pdbs = list(set(test_df['pdb_id'].to_list()))
    test_pdbs = [t.upper()[:4] for t in test_pdbs]
    not_found = []
    for t in train_pdbs:
        if t[:4].upper() not in test_pdbs:
            not_found.append(t)
    
    return not_found

# First and second command line args are the training and test files, respectively. 
# check that train and test files are passed to the script via command line
if len( sys.argv ) > 2:

    train = str(sys.argv[1])
    test = str(sys.argv[2])
     
    root_dir = os.path.dirname(os.path.realpath('__file__'))
    train_path = 'training_data/'
    test_path = 'test_data/'
    train_file = os.path.join(root_dir, train_path + train)
    test_file = os.path.join(root_dir, test_path + test)
    out = 'files/'
    if not os.path.exists(out):
        os.makedirs(out)
    curated = 'curated_data/'
    if not os.path.exists(curated):
        os.makedirs(curated)

    hard_cols = ['wt_seq', 'mut_seq', 'ddg', 'pdb_id', 'pos']

    # training and test initial databases must be in train_data and test_data folders in the root directory
    # check that train and test files exist in the required path
    if os.path.exists(train_file) and os.path.exists(test_file):

        train_df = pd.read_csv(train_file) 
        test_df = pd.read_csv(test_file)

        in_train = check_columns(hard_cols, train_df)
        in_test = check_columns(hard_cols, test_df)

    # check that required columns are present in both datasets and datasets are not empty
    
        if all(x in train_df.columns for x in in_train) and all(x in test_df.columns for x in in_test) \
        and len(train_df) > 1 and len(test_df) > 1:
        
            train_df = parse_columns(train_df)
            test_df = parse_columns(test_df)

         # remove duplicates by 'code'  
            checked = train_df.drop_duplicates('code')
            test_checked = test_df.drop_duplicates('code')
            test_checked.to_csv(curated + test, index=False)

         # remove common proteins shared by train and test sets
            prots_to_keep = non_common_prots(checked, test_df)        
            filter_train = []
            for pdb in prots_to_keep:
                filter_train.append(checked[checked['pdb_id'] == pdb])
        
            df = pd.concat(filter_train)
            df.to_csv(out + test.partition('.')[0] + '-' + train.partition('.')[0] + '_checked', index=False)
    
        else:
            sys.stderr.write('Check that both train and test files are not empty and contain required columns\n')
    else:
        sys.stderr.write('Training or test file not present in expected path (training_data / test_data) \n')
else:
    sys.stderr.write("Training and test files must be command line arguments!\n")
