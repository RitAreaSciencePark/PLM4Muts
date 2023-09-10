#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH -p DGX
#SBATCH --nodes=1
#SBATCH --sockets-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH -o test.%A.out
#SBATCH -e test.%A.error
#SBATCH -A lade
##SBATCH -w dgx002 
#SBATCH --mem=100G

CURRENT_DIR=${SLURM_SUBMIT_DIR}

cd ../..
source myenv_dgx/bin/activate
python src/training_testing.py --current_dir ${CURRENT_DIR} 

cd -


