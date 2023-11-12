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
    
    # Load in datasets
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    wiki_tokenized_dataset = datasets.load_dataset("gmongaras/BERT_Base_Cased_512_Dataset", cache_dir=cache_path)["train"]
    
    # Tokenize the data
    if not os.path.exists(tok_cache_path):
        os.makedirs(tok_cache_path)
    wiki_tokenized_dataset = wiki_tokenized_dataset.map(tokenizer, batched=True, batch_size=1000, num_proc=16, input_columns=["text"], remove_columns=["text"], cache_file_name=tok_cache_path + "/tokenized_dataset.arrow")

    # Push to hub
    wiki_tokenized_dataset.push_to_hub("gmongaras/BERT_Base_Cased_512_Dataset_Mapped", token=TOKEN)
    
    
    
    
if __name__ == "__main__":
    main()
