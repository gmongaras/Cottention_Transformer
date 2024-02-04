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



# Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True, cache_dir="BERT_Trainer/BERT_Model")

# What is the max length for the model?
max_length = tokenizer.model_max_length
# max_length = 128


def clean_text(line):
    # Trim whitespace
    line = line.strip()
    
    # Remove unnecessary characters
    for char_ in ['``', "''", '`` ', "'' "]:
        line = line.replace(char_, '')
        
    # Trim whitespace
    line = line.strip()
    
    # Line should start with alphanumeric character or perenthesis. Remove characters
    # until the first alphanumeric character or perenthesis
    line = re.sub(r'^[^a-zA-Z0-9\(\)]*', '', line).strip()
    
    return line


# Used to get all sentence pairs and tokenize them
def get_pairs(examples):
    # Breakup line into sentences by splitting on periods, but not decimal points
    # and on question marks and exclamation points
    sentences = [clean_text(i) for i in re.split(r'(?<=[.!?])(?<!\d\.\d)(?<!\b[A-Z]\.)(?<!\b(?:e\.g|i\.e|etc|Gen|Maj)\.)(?<!\b(?:No|Lt|vs)\.)(?<!\b(?:Brig|Gens)\.)(?<!\b(?:d)\.)(?=\s|$)(?<!\.\.\.)|\n\n', examples)]
    # sentences = [clean_text(i) for i in re.split(r'(?<!\d)(?<!\b[A-Z])\.(?!\d)|\?|!', examples)]
    
    # Remove sentences longer than the max length and shorter than 3 words
    sentences = [i for i in sentences if len(i.split(" ")) <= max_length//2 and len(i.split(" ")) > 3]
    
    # Add sentence pairs separated by [SEP]
    sentence_pairs = []
    for i in range(len(sentences) - 1):
        sentence_pairs.append({"text": sentences[i] + "[SEP]" + sentences[i + 1]})
        
    return sentence_pairs



# Function to process a batch of examples
def process_batch(batch):
    return [get_pairs(example) for example in batch]

# Function to divide data into batches
def chunk_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]



def main():
    wiki_cache_path = "BERT_Trainer/data_cache/wikipedia_dataset"
    book_cache_path = "BERT_Trainer/data_cache/book_dataset"

    # Load in datasets
    if not os.path.exists(wiki_cache_path):
        os.makedirs(wiki_cache_path)
    if not os.path.exists("BERT_Trainer/wikipedia_dataset"):
        os.makedirs("BERT_Trainer/wikipedia_dataset")
    wiki_data = load_dataset("wikipedia", "20220301.en", split="train", cache_dir="BERT_Trainer/wikipedia_dataset")
    
    
    # Load in datasets
    if not os.path.exists(book_cache_path):
        os.makedirs(book_cache_path)
        
    if not os.path.exists("BERT_Trainer/bookcorpus_dataset"):
        os.makedirs("BERT_Trainer/bookcorpus_dataset")
    book_data = load_dataset("bookcorpus", split="train", cache_dir="BERT_Trainer/bookcorpus_dataset")

    # # Set up basic logging
    # logging.basicConfig(level=logging.INFO)

    # # Determine the chunk size based on dataset size and available CPUs
    # chunk_size = max(1, len(wiki_data) // (2 * cpu_count()))

    # # Process data in batches
    # with ProcessPoolExecutor() as executor:
    #     # Create batches for both datasets
    #     batches = list(itertools.chain(chunk_data(wiki_data["text"], chunk_size), chunk_data(book_data["text"], chunk_size)))

    #     # Total number of batches
    #     total_batches = len(batches)

    #     # Initialize a counter for completed batches
    #     completed_batches = 0

    #     # Process batches in parallel and update the counter
    #     for batch_result in executor.map(process_batch, batches):
    #         completed_batches += 1
    #         logging.info(f"Processed batch {completed_batches}/{total_batches}")

    #         # Extend sentence_pairs with results from the current batch
    #         sentence_pairs.extend(batch_result)

    # # Flattening the list of lists into a single list
    # sentence_pairs = list(itertools.chain.from_iterable(sentence_pairs))
    
    # Collect all sentence pairs from the datasets
    sentence_pairs = []
    for wiki_example in tqdm.tqdm(wiki_data, total=len(wiki_data)):
        sentence_pairs += get_pairs(wiki_example["text"])
            
    for book_example in tqdm.tqdm(book_data, total=len(book_data)):
        sentence_pairs += get_pairs(book_example["text"])




    
    # hf dataset from list
    sentence_pairs = datasets.Dataset.from_list(sentence_pairs)
    
    # Push to hub
    sentence_pairs.push_to_hub(f"gmongaras/BERT_Base_Cased_{max_length}_Dataset", token=TOKEN)
    
    
    
    
if __name__ == "__main__":
    main()
