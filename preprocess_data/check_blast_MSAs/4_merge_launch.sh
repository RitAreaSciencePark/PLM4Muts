#!/bin/bash

#SBATCH --partition=EPYC
#SBATCH --job-name=2RPN-2BTT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=7g
#SBATCH --time=2:00:00
#
srun bash scripts/merge_blast_results.sh "2RPN-2BTT"

echo "finished"
