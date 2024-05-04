# External
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import AutoTokenizer
import shutil
import json
import gc
import os

# Stack Class For Pull, Push
class Stack():
    def __init__(self, cache_dir = "./Stack/models", temp_dir_base = "./Stack", skip_langs = [], tokenizer = "codellama/CodeLlama-7b-hf", 
                 truncation = False, max_length = 1024, batch_size = 1024, num_proc = 16, push_loc = "", token = ""):
        
        # Download Languages JSON
        languages = hf_hub_download(repo_id = "bigcode/the-stack", filename = "programming-languages.json", repo_type = "dataset", cache_dir = cache_dir)

        # Read JSON File
        with open(languages, "r") as file:
            data = json.load(file)

        # Convert Language Names
        for langs in data.keys():
            lang = langs.lower().replace(" ", "-").replace("#", "-sharp")

            # Skip Languages
            if isinstance(skip_langs, list):
                if lang in skip_langs:
                    continue
            if isinstance(skip_langs, str):
                if lang < skip_langs:
                    continue

            # Temporary Storage Directory
            temp_dir = f"{temp_dir_base}/{lang}"
            os.makedirs(temp_dir, exist_ok = True)

            # Load Language Data
            ds = load_dataset("bigcode/the-stack", data_dir = f"data/{lang}", split = "train", cache_dir = temp_dir)

            # Initialize Tokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer, use_fast = True, cache_dir = cache_dir)
            
            # Tokenize Dataset
            tokds = ds.map(lambda examples: tok(examples["content"], truncation = truncation, max_length = max_length), batch_size = batch_size, num_proc = num_proc)

            # Take Certain Information
            remds = tokds.remove_columns([col for col in tokds.column_names if col not in ['input_ids', 'attention_mask']])

            # Push Language Data
            remds.push_to_hub(push_loc, token = token, data_dir = f"data/{lang}", commit_message = f"Tokenized {lang}")

            # Discard Information
            del ds, tokds, remds
            gc.collect()
            shutil.rmtree(temp_dir, ignore_errors = True)