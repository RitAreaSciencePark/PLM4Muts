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

INFILE_TEST_A="datasets/S1465/test/databases/db_ssym.csv"
OUTFILE_TEST_A="datasets/S1465/test/translated_databases/tb_ssym.csv"
echo "Translating $INFILE_TEST_A ..."
OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py ${INFILE_TEST_A} --output_file ${OUTFILE_TEST_A}

#INFILE_TEST_B="datasets/S1465/test/databases/db_ssym_r.csv"
#OUTFILE_TEST_B="datasets/S1465/test/translated_databases/tb_ssym_r.csv"
#echo "Translating $INFILE_TEST_B ..."
#OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py --input_file  ${INFILE_TEST_B} --output_file ${OUTFILE_TEST_B}

#INFILE_TEST_C="datasets/S1465/test/databases/db_ssym_s.csv"
#OUTFILE_TEST_C="datasets/S1465/test/translated_databases/tb_ssym_s.csv"
#echo "Translating $INFILE_TEST_C ..."
#OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py --input_file  ${INFILE_TEST_C} --output_file ${OUTFILE_TEST_C}

INFILE_TEST_D="datasets/S1465/test/databases/db_s669.csv"
OUTFILE_TEST_D="datasets/S1465/test/translated_databases/tb_s669.csv"
echo "Translating $INFILE_TEST_D ..."
OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py  ${INFILE_TEST_D} --output_file ${OUTFILE_TEST_D}

#INFILE_TEST_E="datasets/S1465/test/databases/db_s669_r.csv"
#OUTFILE_TEST_E="datasets/S1465/test/translated_databases/tb_s669_r.csv"
#echo "Translating $INFILE_TEST_E ..."
#OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py --input_file  ${INFILE_TEST_E} --output_file ${OUTFILE_TEST_E}

#INFILE_TEST_F="datasets/S1465/test/databases/db_s669_s.csv"
#OUTFILE_TEST_F="datasets/S1465/test/translated_databases/tb_s669_s.csv"
#echo "Translating $INFILE_TEST_F ..."
#OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py --input_file ${INFILE_TEST_F} --output_file ${OUTFILE_TEST_F}

#INFILE_TEST_G="datasets/S1465/test/databases/db_s669_fix.csv"
#OUTFILE_TEST_G="datasets/S1465/test/translated_databases/tb_s669_fix.csv"
#echo "Translating $INFILE_TEST_G ..."
#OMP_NUM_THREADS=128 torchrun --standalone  --nnodes 1 --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py --input_file ${INFILE_TEST_G} --output_file ${OUTFILE_TEST_G}


