# Cottention Transformer

This repository contains the official implementation of the paper "Cottention: Linear Transformers With Cosine Attention".

## Requirements

To install the necessary dependencies and the custom CUDA kernel, follow these steps:

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Navigate to the CUDA kernel directory and install the kernel:
   ```bash
   cd cuda_kernel
   python -m pip uninstall FastAttention -y
   python setup.py install
   ```

## Training

To train the Cottention model, complete the following:

1. Create dataset with `create_hf_datasets.py`
2. Pre-tokenize dataset with `map_hf_dataset.py`
3. Train model with `train.py`

To run `train.py` with multiple machines, refer to the provided training script. Here's an example of how to run the training using a job scheduler (e.g., SLURM):

```bash
#!/bin/bash

#SBATCH --job-name=training_job
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -o train.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --mem=500G

# No. Nodes
nnodes=1

# No. Tasks Per Node
nproc_per_node=8

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

export LOGLEVEL=INFO

cd /path/to/cottention_transformer

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 srun /path/to/venv/bin/torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
gpt_trainer/train.py
```

Make sure to replace `/path/to/cottention_transformer` with the actual path to your Cottention Transformer repository and `/path/to/venv/bin/torchrun` with the path to your Python virtual environment's `torchrun` executable.

This script sets up the necessary environment variables and launches the training using `torchrun` for distributed training across multiple GPUs.

Note: The provided code assumes you are using a SLURM job scheduler. If you are using a different job scheduler or running the training script directly, you may need to modify the script accordingly.

## Finetuning

To finetune the Cottention model, complete the following:

1. Create dataset with `create_hf_ft_datasets.py`
2. Tokenize dataset with `map_hf_ft_datasets.py`
3. Finetune with `finetune.py`

No different than with training. We have provided all relevant code for BERT. The datasets relevant for GPT will be provided upon release for anonymity.

# Pre-trained Models

You can download pretrained models here:

[Cottention BERT Model](https://drive.google.com/mymodel.pth)  trained on Wikipedia and BookCorpus datasets.
[Cottention GPT Model](https://drive.google.com/mymodel.pth) trained on The Pile dataset.

## Results

Our model achieves the following performance on:

| Model                   | MNLI-(m/mm) | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | Average |
|-------------------------|-------------|------|------|-------|------|-------|------|------|---------|
| BERT_BASE               | 84.6/83.4   | 71.2 | 90.5 | 93.5  | 52.1 | 85.8  | 88.9 | 66.4 | 79.6    |
| BERT_softmax            | 81.8/82.5   | 86.5 | 89.9 | 90.5  | 80.5 | 78.3  | 90.0 | 67.9 | 83.1    |
| BERT_cosine             | 80.6/81.1   | 86.2 | 89.3 | 90.1  | 77.8 | 76.5  | 88.6 | 66.4 | 81.8    |

## Contributing

We welcome contributions to the Cottention Transformer repository. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request. Make sure to follow the established coding style and guidelines.
