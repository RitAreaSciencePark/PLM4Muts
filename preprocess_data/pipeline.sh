#__author__ = "Francesca Cuturello"
#__subject__ = "preprocessing datasets for PLM4Muts training"
#__tags__ = "format, BLASTp, HHblits, filter"
#__copyright__ = "Copyright 2023, AREA Science Park - RIT"
#__credits__ = ["Francesca Cuturello"]
#__license__ = "Creative Commons"
#__version__ = "1.0.0"
#__maintainer__ = "Francesca Cuturello"
#__status__ = "Development"


#!/bin/bash

train=$1
tests=$2

python 1_check_db.py $train $tests

python 2_run_blastp.py $train $tests 

python 3_filter_blast.py $train $tests

python 4_make_MSA.py $train $tests

