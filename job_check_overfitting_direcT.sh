#!/bin/bash
#
#SBATCH --job-name=cutPadri
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
#SBATCH -o cutPadri.%A.out
#SBATCH -e cutPadri.%A.error
#SBATCH -A lade
##SBATCH -w dgx002 
#SBATCH --mem=100G

source myenv_dgx/bin/activate
python training_testing_direcT.py
