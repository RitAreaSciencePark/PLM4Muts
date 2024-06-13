#!/bin/bash
#
#SBATCH --job-name=S1413_Translate
#SBATCH -p boost_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH -o trans.%A.output
#SBATCH -e trans.%A.error
#SBATCH -A cin_staff
#SBATCH --wait-all-nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --exclusive


CURRENT_DIR=${SLURM_SUBMIT_DIR}
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$( srun  --nodes=1 --ntasks=1 -w "$head_node" --exclusive hostname --ip-address)

module purge
module load python
module load cuda
module load nvhpc

cd ../..
echo "head_node=" ${head_node} " - head_node_ip=" $head_node_ip
export OMP_NUM_THREADS=32
export PYTHONPATH=$PYTHONPATH:"$(pwd)/PLM4Muts_venv/lib/python3.11/site-packages/"
source PLM4Muts_venv/bin/activate
echo $(pwd)
echo ${CUDA_VISIBLE_DEVICES}


INFILE_TRAIN="datasets/S1413/train/databases/db_s1413.csv"
OUTFILE_TRAIN="datasets/S1413/train/translated_databases/tb_s1413.csv"
echo "Translating $INFILE_TRAIN ..."
srun -l torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py ${INFILE_TRAIN} --output_file ${OUTFILE_TRAIN}

INFILE_VAL="datasets/S1413/validation/databases/db_ssym.csv"
OUTFILE_VAL="datasets/S1413/validation/translated_databases/tb_ssym.csv"
echo "Translating $INFILE_VAL ..."
srun -l torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py ${INFILE_VAL} --output_file ${OUTFILE_VAL}

INFILE_TEST="datasets/S1413/test/databases/db_s669.csv"
OUTFILE_TEST="datasets/S1413/test/translated_databases/tb_s669.csv"
echo "Translating $INFILE_TEST ..."
srun -l torchrun --nnodes 2 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py ${INFILE_TEST} --output_file ${OUTFILE_TEST}


