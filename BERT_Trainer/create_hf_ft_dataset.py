import transformers
from datasets import load_dataset
import os
import datasets
import tqdm

TOKEN = ""



# Tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased", use_fast=True, cache_dir="BERT_Trainer/BERT_Model")

# What is the max length for the model?
max_length = tokenizer.model_max_length



def main():
    glue_cache_path = "BERT_Trainer/data_cache/glue_dataset"
    dataset_types = [
        ("cola", "single", "sentence"),
        ("mnli_matched", "pair", "premise", "hypothesis"),
        ("mnli_mismatched", "pair", "premise", "hypothesis"),
        ("mrpc", "pair", "sentence1", "sentence2"),
        ("qnli", "pair", "question", "sentence"),
        ("qqp", "pair", "question1", "question2"),
        ("rte", "pair", "sentence1", "sentence2"),
        ("sst2", "single", "sentence"),
        ("stsb", "pair_reg", "sentence1", "sentence2"),
        ("wnli", "pair", "sentence1", "sentence2")
    ]
    
    
    
    # Dataset sentences and labels
    dataset = {
        "train": {
            "sentences": [],
            "labels": [],
            "dataset_name": []
        },
        "validation": {
            "sentences": [],
            "labels": [],
            "dataset_name": []
        },
        "test": {
            "sentences": [],
            "labels": [],
            "dataset_name": []
        }
    }
    
    # Total number of data skipped
    data_skipped = 0
    
    # Iterate over all datasets
    for name, type_, *cols in dataset_types:
        # MNLI is annoying
        if name == "mnli_matched" or name == "mnli_mismatched":
            mnli_type = "matched" if name == "mnli_matched" else "mismatched"
            name = "mnli"
        
        
        # Load in dataset
        if not os.path.exists(glue_cache_path):
            os.makedirs(glue_cache_path)
        if not os.path.exists("BERT_Trainer/glue_dataset"):
            os.makedirs("BERT_Trainer/glue_dataset")
        glue_data = load_dataset("glue", name, cache_dir="BERT_Trainer/glue_dataset")
        
        # Iterate over all splits
        for split in ["train", "validation", "test"]:
            print(f"Dataset: {name}, Split: {split}")
            
            # Skip MNLI if the train data is seen a second time
            if "mnli" in dataset["train"]["dataset_name"] and name == "mnli" and split == "train":
                continue
            
            # Save sentence depending on the type
            for example in tqdm.tqdm(glue_data[split + "_" + mnli_type if name=="mnli" and split != "train" else split], total=len(glue_data[split + "_" + mnli_type if name=="mnli" and split != "train" else split])):
                if type_ == "single":
                    sentence = example[cols[0]]
                elif type_ == "pair" or type_ == "pair_reg":
                    sentence = example[cols[0]] + "[SEP]" + example[cols[1]]
                else:
                    raise ValueError(f"Unknown type {type_}")
                
                # Ensure sentence is not too long
                if len(tokenizer(sentence)["input_ids"]) > max_length:
                    data_skipped += 1
                    continue
                
                dataset[split]["sentences"].append(sentence)
                dataset[split]["labels"].append(example["label"])
                if (split == "validation" or split == "test") and name == "mnli":
                    dataset[split]["dataset_name"].append(name + "_" + mnli_type + "_" + split)
                else:
                    dataset[split]["dataset_name"].append(name)




    
    # Convert the dataset splits to datasets
    sentence_pairs = datasets.DatasetDict({
        split: datasets.Dataset.from_dict({
            "sentence": dataset[split]["sentences"],
            "label": dataset[split]["labels"],
            "dataset_name": dataset[split]["dataset_name"]
        })
        for split in dataset
    })
    
    # Push to hub
    sentence_pairs.push_to_hub("gmongaras/BERT_Base_Cased_512_GLUE", token=TOKEN)
    
    print(f"Skipped {data_skipped} datapoints")
    
    
    
    
if __name__ == "__main__":
    main()
