from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import itertools
import pandas as pd
import os

prot = "1A23"
filename_msa = "../preprocess_PLM4Muts/S1465/train/MSA_S1465/1A23" 

def read_msa(prot, filename_msa):
    
    max_tokens = 16000
    records_msa = list(SeqIO.parse(filename_msa,  "fasta"))
    
    lmsa = len(records_msa)
    lseq = max([len(records_msa[i].seq) for i in range(lmsa)]) 
    assert lseq < 1024
    assert 2 * lseq + 2 < max_tokens
    nseqs = int(max_tokens//(lseq + 1))
    nseqs = min(nseqs, lmsa) 
    idx = list(range(0, nseqs)) 
    
    file = open('small_used_msa/'+ prot + '_used','w')
    for i in idx:
        file.write(">" + records_msa[i].description + '\n')
        file.write(str(records_msa[i].seq + '\n'))
    file.close()

read_msa(prot, filename_msa)
