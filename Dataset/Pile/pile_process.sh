#!/bin/bash

#SBATCH -J ILYHOWL!
#SBATCH -p standard-s
#SBATCH --array=0-63
#SBATCH --exclusive

# Load Python Environment, Execute
cd ../Cottention_Transformer/Dataset/Pile
python pilesubset.py --job-index ${SLURM_ARRAY_TASK_ID} --total-jobs 64

