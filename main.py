import torch
import random
from Transformer import Transformer
from datasets import load_dataset, load_from_disk






def main():
    # Model params
    dim = 128
    num_layers = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training params
    batch_size = 128
    learning_rate = 1e-4
    epochs = 1000
    max_length = 200
    
    
    
    
    # # Load in the text datasets
    # with open(english_path, "r", encoding="utf-8") as f:
    #     english = f.readlines()
    # with open(spanish_path, "r", encoding="utf-8") as f:
    #     spanish = f.readlines()
    
    # # Combine lists
    # dataset = list(zip(english, spanish))
    
    # # Cut dataset to be an even multiple of batch size
    # dataset = dataset[:len(dataset) - (len(dataset) % batch_size)]
    
    
    
    
    # Create the model
    model = Transformer(num_layers, dim).to(device)
    
    
    
    
    
    
    # # Load in dataset
    # from datasets import load_dataset, load_from_disk
    # dataset = load_dataset(
    #     "gmongaras/reddit_political_2019_Feb", 
    #     cache_dir="./datasets",
    # )
    
    # # Load in the dataset and map using the tokenizer
    # def map_function(example):
    #     text = example["text"]
        
    #     # Encode the question and output
    #     text_encoded = model.tokenizer(text, max_length=max_length-1, truncation=True, padding="max_length")
        
    #     # Add on a pad token to the end of the input_ids
    #     text_encoded["input_ids"] = text_encoded["input_ids"] + [model.tokenizer.pad_token_id]
        
    #     # Attention mask is the length of the input_ids without the padding + 1
    #     # because we want the model to stop itself
    #     attention_mask = [1 for i in range(0, sum(text_encoded["attention_mask"]) + 1)] + [0 for i in range(sum(text_encoded["attention_mask"])+1, max_length)]
    #     assert len(attention_mask) == max_length and len(text_encoded["input_ids"]) == max_length, \
    #         "Attention mask or input_ids is not the correct length"
    #     # attention_mask = text_encoded["attention_mask"]
        
    #     # The labels are the input ids, but we want to mask the loss for the context and padding
    #     labels = [text_encoded["input_ids"][i] if attention_mask[i] == 1 else -100 for i in range(len(attention_mask))]
    #     assert len(labels) == len(attention_mask) and len(attention_mask) == len(text_encoded["input_ids"]), "Labels is not the correct length"
        
    #     return {
    #         "input_ids": text_encoded["input_ids"],
    #         "labels": labels,
    #         "attention_mask": attention_mask
    #     }
    
    # # Take a subset of 1000
    # dataset = dataset["train"].select(range(1000000))
    
    # # Map the dataset
    # dataset = dataset.map(map_function, batched=False)
    
    # # Remove text from dataset
    # # dataset = dataset.remove_columns(["text"])["train"]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    from datasets import load_dataset
    
    # Load in the dataset and map using the tokenizer
    def map_function(example):
        text = example["text"]
        
        # Encode the question and output
        text_encoded = model.tokenizer(text, max_length=max_length-1, truncation=True, padding="max_length")
        
        # Add on a pad token to the end of the input_ids
        text_encoded["input_ids"] = text_encoded["input_ids"] + [model.tokenizer.pad_token_id]
        
        # Attention mask is the length of the input_ids without the padding + 1
        # because we want the model to stop itself
        attention_mask = [1 for i in range(0, sum(text_encoded["attention_mask"]) + 1)] + [0 for i in range(sum(text_encoded["attention_mask"])+1, max_length)]
        assert len(attention_mask) == max_length and len(text_encoded["input_ids"]) == max_length, \
            "Attention mask or input_ids is not the correct length"
        # attention_mask = text_encoded["attention_mask"]
        
        # The labels are the input ids, but we want to mask the loss for the context and padding
        labels = [text_encoded["input_ids"][i] if attention_mask[i] == 1 else -100 for i in range(len(attention_mask))]
        assert len(labels) == len(attention_mask) and len(attention_mask) == len(text_encoded["input_ids"]), "Labels is not the correct length"
        
        return {
            "input_ids": text_encoded["input_ids"],
            "labels": labels,
            "attention_mask": attention_mask
        }
    
    
    dataset = load_dataset(
        "c4", 
        name="realnewslike",
        cache_dir="datasets",
    )["train"]
    dataset = dataset.select(range(250000))
    dataset = dataset.map(map_function, batched=False)
    
    
    
    
    
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    
    
    
    for epoch in range(0, epochs):
        # Shuffle the dataset
        dataset = dataset.shuffle()
        
        # Iterate over all batches
        for i in range(0, len(dataset), batch_size):
            # Get the batch
            batch = dataset[i:i+batch_size]
            
            batch["input_ids"] = torch.tensor(batch["input_ids"]).to(device)
            batch["labels"] = torch.tensor(batch["labels"]).to(device)
            batch["attention_mask"] = torch.tensor(batch["attention_mask"]).to(device)
            
            # # Get the english and spanish sentences
            # english = [sentence[0] for sentence in batch]
            # spanish = [sentence[1] for sentence in batch]
            
            # # Tokenize the english and spanish sentences
            # english = model.tokenizer(english, padding=True, return_tensors="pt")
            # spanish = model.tokenizer(spanish, padding=True, return_tensors="pt")
            
            # Send through model
            output = model(batch)
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(output.transpose(-1, -2), batch["labels"])
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(loss.item())
            
            
            
            
            
            
if __name__ == "__main__":
    main()