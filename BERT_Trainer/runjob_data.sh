#!/bin/sh
#SBATCH --gres=gpu:0

cd /users/gmongaras/work/Cottention_Transformer
python BERT_Trainer/create_hf_datasets.py