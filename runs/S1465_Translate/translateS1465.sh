#!/bin/bash
#
#SBATCH --job-name=translateS1465
#SBATCH -p DGX
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH -o trans.%A.out
#SBATCH -e trans.%A.error
#SBATCH -A lade
#SBATCH --wait-all-nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=900G
#SBATCH --exclusive 
CURRENT_DIR=${SLURM_SUBMIT_DIR}
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$( srun  --nodes=1 --ntasks=1 -w "$head_node" --exclusive hostname --ip-address)
echo "head_node=" ${head_node} " - head_node_ip=" $head_node_ip
export OMP_NUM_THREADS=128
cd ../..
source PLM4Muts_venv/bin/activate
echo $(pwd)
echo ${CUDA_VISIBLE_DEVICES}

INFILE_TRAIN="datasets/S1465/train/databases/db_s1465.csv"
OUTFILE_TRAIN="datasets/S1465/train/translated_databases/tb_s1465.csv"
echo "Translating $INFILE_TRAIN ..."
OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py ${INFILE_TRAIN} --output_file ${OUTFILE_TRAIN}

INFILE_VAL="datasets/S1465/validation/databases/db_ssym.csv"
OUTFILE_VAL="datasets/S1465/validation/translated_databases/tb_ssym.csv"
echo "Translating $INFILE_VAL ..."
OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py ${INFILE_VAL} --output_file ${OUTFILE_VAL}

INFILE_TEST="datasets/S1465/test/databases/db_s669.csv"
OUTFILE_TEST="datasets/S1465/test/translated_databases/tb_s669.csv"
echo "Translating $INFILE_TEST ..."
OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py ${INFILE_TEST} --output_file ${OUTFILE_TEST}


