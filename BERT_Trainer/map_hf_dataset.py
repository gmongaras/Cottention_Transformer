import transformers
from datasets import load_dataset
import os
import re
import datasets
import tqdm

from concurrent.futures import ProcessPoolExecutor
import itertools
from multiprocessing import cpu_count
import logging

TOKEN = ""





def main():
    # Cache dirs
    cache_path = "BERT_Trainer/data_cache/dataset"
    tok_cache_path = "BERT_Trainer/data_cache/tokenized_dataset"
    
    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True, cache_dir="BERT_Trainer/BERT_Model")

    # What is the max length for the model?
    max_length = tokenizer.model_max_length
    # max_length = 128
    
    # Load in datasets
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    wiki_tokenized_dataset = datasets.load_dataset(f"gmongaras/BERT_Base_Cased_{max_length}_Dataset", cache_dir=cache_path)["train"]
    
    # Tokenize the data
    if not os.path.exists(tok_cache_path):
        os.makedirs(tok_cache_path)
    wiki_tokenized_dataset = wiki_tokenized_dataset.map(tokenizer, batched=True, batch_size=1000, num_proc=16, input_columns=["text"], remove_columns=["text"], cache_file_name=tok_cache_path + "/tokenized_dataset.arrow")
    
    # Remove sentences longer than the max length and shorter than 3 words
    wiki_tokenized_dataset = wiki_tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    # Push to hub
    wiki_tokenized_dataset.push_to_hub(f"gmongaras/BERT_Base_Cased_{max_length}_Dataset_Mapped", token=TOKEN)
    
    
    
    
if __name__ == "__main__":
    main()
