import transformers
from datasets import load_dataset
import os

TOKEN = "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"



def main():
    wiki_cache_path = "BERT_Trainer/data_cache/wikipedia_dataset"
    book_cache_path = "BERT_Trainer/data_cache/book_dataset"
    
    
    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False, cache_dir="BERT_Trainer/BERT_Model")

    # BERT Model
    model = transformers.BertModel.from_pretrained("bert-base-cased", cache_dir="BERT_Trainer/BERT_Model", output_hidden_states=True, output_attentions=True)


    # Function to tokenize the "text" column of the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=model.config.max_position_embeddings)

    # Load in datasets
    if not os.path.exists(wiki_cache_path):
        os.makedirs(wiki_cache_path)
    # wiki_data = load_dataset("wikipedia", language="en", date="20220301.en", split="train", cache_dir="BERT_Trainer/wikipedia_dataset", beam_runner='DirectRunner')
    if not os.path.exists("BERT_Trainer/wikipedia_dataset"):
        os.makedirs("BERT_Trainer/wikipedia_dataset")
    wiki_data = load_dataset("wikipedia", "20220301.en", split="train", cache_dir="BERT_Trainer/wikipedia_dataset")
    
    # Tokenize the dataset
    wiki_tokenized_dataset = wiki_data.map(tokenize_function, batched=True, num_proc=10, remove_columns=["text", "id", "url", "title"], cache_file_name=wiki_cache_path + "/wikipedia_tokenized_dataset.arrow")
    
    # # Load dataset
    # wiki_tokenized_dataset = load_dataset(wiki_cache_path)
    
    # Upload to huggingface hub
    wiki_tokenized_dataset.push_to_hub("gmongaras/wikipedia_BERT_512", token=TOKEN)
    
    
    # Load in datasets
    if not os.path.exists(book_cache_path):
        os.makedirs(book_cache_path)
        
    if not os.path.exists("BERT_Trainer/bookcorpus_dataset"):
        os.makedirs("BERT_Trainer/bookcorpus_dataset")
    book_data = load_dataset("bookcorpus", split="train", cache_dir="BERT_Trainer/bookcorpus_dataset")

    # Tokenize the dataset
    book_tokenized_dataset = book_data.map(tokenize_function, batched=True, num_proc=10, remove_columns=["text"], cache_file_name=book_cache_path + "/book_tokenized_dataset.arrow", load_from_cache_file=True)
    
    # Load dataset
    # book_tokenized_dataset = load_dataset(book_cache_path, keep_in_memory=False)
        
    # Upload to huggingface hub
    book_tokenized_dataset.push_to_hub("gmongaras/book_BERT_512", token=TOKEN, num_shards=10)
    
    
    
    
if __name__ == "__main__":
    main()
