import torch
import datasets
import os


TOKEN = ""




def main():
    # Cache dirs
    wiki_cache_path = "BERT_Trainer/data_cache/wikipedia_dataset"
    book_cache_path = "BERT_Trainer/data_cache/book_dataset"
    
    # Load in datasets
    if not os.path.exists(wiki_cache_path):
        os.makedirs(wiki_cache_path)
    wiki_tokenized_dataset = datasets.load_dataset("gmongaras/wikipedia_BERT_512", cache_dir=wiki_cache_path)["train"]
    
    if not os.path.exists(book_cache_path):
        os.makedirs(book_cache_path)
    book_tokenized_dataset = datasets.load_dataset("gmongaras/book_BERT_512", cache_dir=book_cache_path)["train"]
    
    # Merge datasets on "text" column
    merged_dataset = datasets.concatenate_datasets([wiki_tokenized_dataset, book_tokenized_dataset], )
    
    # Push to hub
    merged_dataset.push_to_hub("gmongaras/wikipedia_book_BERT_512", token=TOKEN, num_shards=20)





if __name__ == "__main__":
    main()