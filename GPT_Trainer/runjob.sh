#!/bin/bash

#SBATCH --job-name=Nya~GPT~^w^
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -o runjob.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=500G


# Number of nodes
nnodes=1
# Number of tasks per node
nproc_per_node=8



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

cd /users/gmongaras/work/Cottention_Transformer
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 srun /home/gmongaras/miniconda3/bin/torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
GPT_Trainer/train.py