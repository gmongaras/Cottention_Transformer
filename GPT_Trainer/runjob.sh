#!/bin/bash

#SBATCH --job-name=Nya~GPT~^w^
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -o runjob.out
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --mem=500G

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

cd /users/gmongaras/work/Cottention_Transformer
srun /home/gmongaras/miniconda3/bin/torchrun \
--nnodes 2 \
--nproc_per_node 8 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
GPT_Trainer/train.py