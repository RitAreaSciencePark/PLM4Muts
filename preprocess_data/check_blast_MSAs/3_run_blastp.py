import numpy as np
import pandas as pd
import subprocess
import sys
import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

root_dir = os.path.dirname(os.path.realpath('__file__'))

test = "3BCI"
test_msa = "s669_used_msa/"+test+"_used"
train = "1A23"
train_msa = "small_used_msa/"+train+"_used"

dir_code = test + '-' + train
if not os.path.exists('blastp/' + dir_code):
    os.makedirs('blastp/' + dir_code)

def read_msa(prot, filename_msa):

    records_msa = list(SeqIO.parse(filename_msa,  "fasta"))

    return records_msa

        
records_test = read_msa(test, test_msa)
test_nseqs = len(records_test)
records_train = read_msa(train, train_msa)
train_nseqs = len(records_train)

print(test_nseqs, train_nseqs)
for i in range(test_nseqs):

    test_seq = str(records_test[i].seq)

    for j in range(train_nseqs):

        train_seq = str(records_train[j].seq)

        file_code = test + "_"+ str(i) + "-" + train + "_" + str(j)

        subprocess.check_output(['bash','rugge_to_sbatch.sh', test_seq, train_seq, dir_code, file_code]).decode().split()
         
#        subprocess.check_output(['bash', 'scripts/merge_blast_results.sh', dir_code])


