export OMP_NUM_THREADS=64
cd ../..
source myenv_esm/bin/activate
echo $(pwd)
echo ${CUDA_VISIBLE_DEVICES}

INFILE_TRAIN="datasets/standard/train/MSA_databases/db_train.csv"
OUTFILE_TRAIN="datasets/standard/train/MSA_3Di_databases/tb_train.csv"

INFILE_TEST="datasets/standard/test/MSA_databases/db_s669.csv"
OUTFILE_TEST="datasets/standard/test/MSA_3Di_databases/tb_s669.csv"

#OMP_NUM_THREADS=64 torchrun --standalone  --nnodes 1 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py --input_file  ${INFILE_TRAIN} --output_file ${OUTFILE_TRAIN}

OMP_NUM_THREADS=64 torchrun --standalone  --nnodes 1 --nproc_per_node 4 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 src/ProstT5TranslationDDP.py --input_file  ${INFILE_TEST} --output_file ${OUTFILE_TEST}
