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

MODEL="Padriciano"
LR=1e-5
OPTIMIZER="Adam"
MAX_EPOCHS=1
CURRENT_DIR=${SLURM_SUBMIT_DIR}
TRAIN_DIR="datasets/train/train_example"
VAL_DIR="datasets/val/val_example"
LOSS_FN="MSE"
DEVICE="cuda"
VERBOSE="True"

cd ../..
source myenv_dgx/bin/activate
python src/training_testing.py --model ${MODEL} --lr ${LR} --optimizer ${OPTIMIZER} \
	                       --max_epochs ${MAX_EPOCHS} --current_dir ${CURRENT_DIR} --loss_fn ${LOSS_FN} \
			       --train_dir ${TRAIN_DIR} --val_dir ${VAL_DIR} --device ${DEVICE} \
	                       --verbose ${VERBOSE}

cd -
