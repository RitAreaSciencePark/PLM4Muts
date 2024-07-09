# SPDX-FileCopyrightText: 2024 (C) 2024 Francesca Cuturello <francesca.cuturello@areasciencepark.it>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import subprocess
import sys
import os

root_dir = os.path.dirname(os.path.realpath('__file__'))
train_path = 'files/'
test_path = 'test_data/'

if len(sys.argv) > 2: 
    
    train = str(sys.argv[1]).partition('.')[0]
    test = str(sys.argv[2]).partition('.')[0]
    parsed = test +'-'+ train+'_checked'
    train_file = os.path.join(root_dir, train_path + parsed)
    test_file = os.path.join(root_dir, test_path + str(sys.argv[2]))
    
    if os.path.exists(train_file) and os.path.exists(train_file):
        
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

        dir_code = test +'-'+ train 
        if not os.path.exists('blastp/' + dir_code):
            os.makedirs('blastp/' + dir_code)
        
        # Loop on test and training sequences. Each pair is input of bash script that launches blastp alignment  
        for id_test in list(set(test_df['pdb_id'].to_list())):

            test_seq = test_df[test_df['pdb_id'] == id_test]['wt_seq'].values[0]

            for id_train in list(set(train_df['pdb_id'].to_list())):
        
                train_seq = train_df[train_df['pdb_id'] == id_train]['wt_seq'].values[0]

                file_code = id_test.upper() + '-' + id_train.upper()

                subprocess.check_output(['bash','scripts/blast.sh', test_seq, train_seq, dir_code, file_code]).decode().split()
         
        subprocess.check_output(['bash', 'scripts/merge_blast_results.sh', dir_code])

    else:
        sys.stderr.write('Training or test file not present in expected path (files / test_data) \n')
else:
    sys.stderr.write("Training and test files must be command line arguments!\n")

