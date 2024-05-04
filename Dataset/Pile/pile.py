# Externals
from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer
import shutil
import os
import gc

# Pile Class For Pull, Push
class Pile():
    def __init__(self, process = True, push = True, cache_dir = "./Pile/models", data_dir = "./Pile/data", tokenizer = "meta-llama/Llama-2-7b-hf", 
                 job_index = -1, total_jobs = 64, truncation = False, max_length = 1024, batch_size = 1024, num_proc = 16, num_shards = 200, push_loc = "", token = ""):

        # One Core: Process, Save Each Subset Individually (Job Index -1)
        # Multiple Cores: Parallelized Processing, Saving Of Subsets (Can Use Process Scheduler For This)
        if process:
            if job_index == -1:
                for i in range(total_jobs):
                    self.process_and_save_subset(cache_dir, data_dir, tokenizer, i, total_jobs, truncation, max_length, batch_size, num_proc)
            else:
                self.process_and_save_subset(cache_dir, data_dir, tokenizer, job_index, total_jobs, truncation, max_length, batch_size, num_proc)
        
        # Combine All Subsets Into Single Dataset, Push To Hugging Face Hub
        if push:
            self.combine_and_push_datasets(data_dir, push_loc, num_shards, token)
        
    def process_and_save_subset(self, cache_dir, data_dir, tokenizer, job_index, total_jobs, truncation, max_length, batch_size, num_proc):

        # Load Dataset Split
        os.makedirs(data_dir, exist_ok = True)
        dataset = load_dataset("gmongaras/EleutherAI_the_pile_deduplicated", split = "train", cache_dir = data_dir)

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
        
        # Initialize Tokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer, use_fast = True, cache_dir = cache_dir)
        
        # Tokenize Dataset
        tokds = subset.map(lambda examples: tok(examples["text"], truncation = truncation, max_length = max_length), batch_size = batch_size, num_proc = num_proc)
        
        # Take Certain Information
        remds = tokds.remove_columns([col for col in tokds.column_names if col not in ['input_ids', 'attention_mask']])

        # Save Processed Subset Locally
        remds.save_to_disk(f"{data_dir}/processed_subset_{job_index}")

        # Discard Information
        del tokds, remds
        gc.collect()
        
    def combine_and_push_datasets(self, data_dir, push_loc, num_shards, token):
        
        # Automatically Detect Subset Directories
        subset_dirs = sorted([d for d in os.listdir(data_dir) if d.startswith("processed_subset_")])
        subsets = []
        
        # Load All Detected Subsets
        for subset_dir in subset_dirs:
            subset_path = os.path.join(data_dir, subset_dir)
            subset = load_from_disk(subset_path)
            subsets.append(subset)
            
        # Combine Subsets Into Single Dataset
        combined_dataset = concatenate_datasets(subsets)
        
        # Push To Hugging Face Hub
        combined_dataset.push_to_hub(push_loc, num_shards = num_shards, token = token)

        # Remove Dataset (Can Remove If Necessary)
        shutil.rmtree(data_dir, ignore_errors = True)
