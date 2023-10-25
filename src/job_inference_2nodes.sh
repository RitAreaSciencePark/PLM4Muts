#!/bin/bash
#
#SBATCH --job-name=multinode-example
#SBATCH -p DGX
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH -o test.%A.out
#SBATCH -e test.%A.error
#SBATCH -A lade
#SBATCH --wait-all-nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G 
##SBATCH -w dgx002
CURRENT_DIR=${SLURM_SUBMIT_DIR}
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
head_node_ip=$( srun  --nodes=1 --ntasks=1 -w "$head_node" --exclusive hostname --ip-address)
#head_node_ip=10.128.6.161
echo "head_node=" ${head_node} " - head_node_ip=" $head_node_ip
#export LOGLEVEL=INFO
#export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=64
cd ..
source myenv_Lucrezia/bin/activate
echo $(pwd)
echo ${CUDA_VISIBLE_DEVICES}
cd -
srun -l torchrun   \
--nnodes 2 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
Marco_data_distributed_inference.py 


