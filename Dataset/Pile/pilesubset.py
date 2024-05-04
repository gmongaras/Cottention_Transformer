# Externals
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import os
import gc

# Just Process, Save - No Push
# For Example With SLURM Script
def process_and_save_subset(job_index, total_jobs):
    
    # Load Dataset Split
    os.makedirs("./Pile/data", exist_ok = True)
    dataset = load_dataset("gmongaras/EleutherAI_the_pile_deduplicated", split = "train", cache_dir = "./Pile/data")

    # Manually Calculate Subset Indices For Sharding
    total_size = len(dataset)
    per_job = total_size // total_jobs
    start = job_index * per_job
    end = (job_index + 1) * per_job if job_index < total_jobs - 1 else total_size
    subset = dataset.select(range(start, end))

    # Filter Out Sequences Longer Than 500K Characters
    subset = subset.filter(lambda x: len(x["text"]) <= 500000)

    # Clear Memory
    del dataset
    gc.collect()

    # Initialize Tokenizer (https://huggingface.co/docs/transformers/main_classes/tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", use_fast = True, cache_dir = "./")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast = True, cache_dir = "./")

    # Tokenize Dataset
    tokenized_dataset = subset.map(lambda examples: tokenizer(examples["text"], truncation = False), batch_size = 1024, num_proc = 64)

    # Save Processed Subset Locally
    tokenized_dataset.save_to_disk(f"./Pile/data/processed_subset_{job_index}.arrow")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-index", type = int, required = True, help = "Index Of Job Within Array")
    parser.add_argument("--total-jobs", type = int, required = True, help = "Total No. Jobs Within Array")
    args = parser.parse_args()
    process_and_save_subset(args.job_index, args.total_jobs)
