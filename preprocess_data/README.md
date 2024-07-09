<!--
SPDX-FileCopyrightText: 2024 (C) 2024 Francesca Cuturello <francesca.cuturello@areasciencepark.it>

SPDX-License-Identifier: CC-BY-4.0
-->

# README.md

# Requirements

## Packages

- BLAST 2.15.0+
	```
	wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.15.0+-x64-linux.tar.gz
	tar zxvpf ncbi-blast-2.15.0+-x64-linux.tar.gz
	export PATH="$(pwd)/ncbi-blast-2.10.1+/bin:$PATH"
	```

- HHblits 3.3.0
	```
	git clone https://github.com/soedinglab/hh-suite.git
	mkdir -p hh-suite/build && cd hh-suite/build
	cmake -DCMAKE_INSTALL_PREFIX=. ..
	make -j 4 && make install
	export PATH="$(pwd)/bin:$(pwd)/scripts:$PATH"
	```
- uniclust30_2018_08 database in the current directory. It will be downloaded authomatically if not found in place.

## Input data format

Training and test sets files must be located in the ./training_data and ./test_data directories. Datasets must be in csv format and contain the following columns:
**'pdb_id', 'mut_info', 'pos', 'wt_seq', 'mut_seq', 'ddg'**

# Run

`bash pipeline.sh train.csv test.csv`

## Pipeline flow

**1_check_db.py**

    checks datasets format and excludes pdb from train.csv if present in test.csv
    output: ./files/test-train_checked 
            ./curated_data/test.csv

**2_run_blastp.py**

    pairwise Blastp alignment between training and test sequences

    subprocesses:
    - blast.sh
    - merge_blast_results.sh

    output: ./blastp/test-train_merged_blast

**3_filter_blast.py**

    exclude training sequences overlapping with test sequences
    output: ./files/train_for_test

**4_make_MSA.py**

    builds Multiple Sequence Alignments searchin wild types in UniClust30 database with HHblits

    subprocesses:
    - hhblits_search.sh
    - check_msa.sh
    - parse_msa.sh

    output: ./curated_data/train_for_test-clean.csv
            ./hhblits_files/train/ and ./hhblits_files/test containing formatted wild type and mutate MSA

# Final data structure

Merge training sets built for two different test sets, test1 and test2, and format for training the model:

`bash make_final_datasets.sh train_for_test1-clean.csv train_for_test2-clean.csv test1.csv validation.csv`

For one test set, command line must be redundant:

`bash make_final_datasets.sh train_for_test-clean.csv train_for_test-clean.csv test.csv validation.csv`

subprocess: **5_dbs_merge.py**
