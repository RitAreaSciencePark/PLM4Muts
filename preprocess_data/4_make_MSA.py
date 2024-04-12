import pandas as pd
import sys
import os
import subprocess


train = str(sys.argv[1]).partition('.')[0]
test = str(sys.argv[2]).partition('.')[0]

train_name = train + '_for_' + test 
train_filtered = pd.read_csv('files/' + train_name)
test_df = pd.read_csv('curated_data/' + str(sys.argv[2]))

# TRAINING
# make training query sequence file
qfolder_train = 'hhblits_files/queries/'+ train + '/'
if not os.path.exists(qfolder_train):
    os.makedirs(qfolder_train)

unique_pdbs_train = list(set(train_filtered['pdb_id'].to_list()))
for name in unique_pdbs_train:
    file = open(qfolder_train + '/' + name.replace('.pdb',''), 'w')
    seq = str(train_filtered[train_filtered['pdb_id'] == name]['wt_seq'].values[0])
    file.write('>' + name +'\n')
    file.write(seq + '\n')
    file.close()

# search of query homologs with hhblits on UniClust30
subprocess.check_output(['bash','scripts/hhblits_search.sh', train])
# check that training msa are not empty
subprocess.check_output(['bash','scripts/check_msa.sh', train, test])
# format MSA and create mutated MSA
subprocess.check_output(['bash','scripts/parse_msa.sh', train, test, "train"])


# TEST
# make test query sequence file
qfolder_test = 'hhblits_files/queries/'+ test + '/'
if not os.path.exists(qfolder_test):
    os.makedirs(qfolder_test)

unique_pdbs_test = list(set(test_df['pdb_id'].to_list()))
for name in unique_pdbs_test:
    file = open(qfolder_test + '/' + name.replace('.pdb',''), 'w')
    seq = str(test_df[test_df['pdb_id'] == name]['wt_seq'].values[0])
    file.write('>' + name +'\n')
    file.write(seq + '\n')
    file.close()

# search of query homologs with hhblits on UniClust30
subprocess.check_output(['bash','scripts/hhblits_search.sh', test])
# format MSA and create mutated MSA
subprocess.check_output(['bash','scripts/parse_msa.sh', train, test, "test"])
