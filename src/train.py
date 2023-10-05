import torch
import random
from datasets import load_dataset, load_from_disk
import os
import wandb


try:
    import sys
    sys.path.append('src')
    sys.path.append('src/Model')
    
    from Trainer import Trainer
    from Transformer import Transformer
except ModuleNotFoundError:
    from src.Trainer import Trainer
    from src.Model.Transformer import Transformer






def main():
    # Model params
    dim = 512
    num_layers = 15
    scale_factor = 2
    distance_type = "cosine"
    activation_type = "relu"
    # distance_type = "l2"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "gpu"
    
    # Training params
    batch_size = 256
    learning_rate = 1e-4
    epochs = 1000
    max_length = 200
    num_workers = 10
    prefetch_factor = 16
    save_every_steps = 1000
    use_scheduler = True
    checkpoints_dir = "checkpoints"
    optimizer_checkpoint = None
    accumulation_steps = 1
    scheduler_checkpoint = None
    use_amp = True
    clipping_value = 1.0
    weight_decay = 0.1
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    warmup_steps = 1000
    wandb_name = None
    test_per = 0.1
    
    
    
    
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
    model = Transformer(num_layers=num_layers, 
        dim=dim, 
        scale_factor=scale_factor, 
        distance_type=distance_type,
        activation_type=activation_type,
    )
    
    
    
    
    
    
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
    #     text_encoded = model.tokenizer[0](text, max_length=max_length-1, truncation=True, padding="max_length")
        
    #     # Add on a pad token to the end of the input_ids
    #     text_encoded["input_ids"] = text_encoded["input_ids"] + [model.tokenizer[0].pad_token_id]
        
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
    # dataset = dataset["train"].select(range(1000))
    
    # # Map the dataset
    # dataset = dataset.map(map_function, batched=False)
    
    # # Remove text from dataset
    # # dataset = dataset.remove_columns(["text"])["train"]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    from datasets import load_dataset
    
    # Load in the dataset and map using the tokenizer
    def map_function(example):
        text = example["text"]
        
        # Encode the question and output
        text_encoded = model.tokenizer[0](text, max_length=max_length-1, truncation=True, padding="max_length")
        
        # Add on a pad token to the end of the input_ids
        text_encoded["input_ids"] = text_encoded["input_ids"] + [model.tokenizer[0].pad_token_id]
        
        # Attention mask is the length of the input_ids without the padding + 1
        # because we want the model to stop itself
        attention_mask = [1 for i in range(0, sum(text_encoded["attention_mask"]) + 1)] + [0 for i in range(sum(text_encoded["attention_mask"])+1, max_length)]
        assert len(attention_mask) == max_length and len(text_encoded["input_ids"]) == max_length, \
            "Attention mask or input_ids is not the correct length"
        # attention_mask = text_encoded["attention_mask"]
        
        # The labels are the input ids shifted by 1, but we want to mask the loss for the context and padding
        labels = [text_encoded["input_ids"][i] if attention_mask[i] == 1 else -100 for i in range(len(attention_mask))]
        assert len(labels) == len(attention_mask) and len(attention_mask) == len(text_encoded["input_ids"]), "Labels is not the correct length"
        
        # Shift the labels by 1
        labels = labels[1:] + [-100]
        
        return {
            "input_ids": text_encoded["input_ids"],
            "labels": labels,
            "attention_mask": attention_mask
        }
    
    # Take a subset of 1000 on loading
    dataset = load_dataset(
        "c4", 
        name="realnewslike",
        cache_dir="datasets",
    )["train"]
    load_from_cache_file = True if os.path.exists("realnewslike") else False
    dataset = dataset.map(map_function, batched=False, load_from_cache_file=load_from_cache_file, cache_file_name="realnewslike")
    
    
    
    # Model trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        dev=device,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        lr=learning_rate,
        save_every_steps=save_every_steps,
        use_scheduler=use_scheduler,
        checkpoints_dir=checkpoints_dir,
        optimizer_checkpoint=optimizer_checkpoint,
        accumulation_steps=accumulation_steps,
        scheduler_checkpoint=scheduler_checkpoint,
        use_amp=use_amp,
        clipping_value=clipping_value,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        warmup_steps=warmup_steps,
        wandb_name=wandb_name,
        test_per=test_per,
    )
    
    # Train model
    trainer.train()
    
    exit()
    
    
    
    
    for epoch in range(0, epochs):
        # Shuffle the dataset
        dataset = dataset.shuffle()
        
        # Iterate over all batches
        # for i in range(0, len(dataset), batch_size):
        for i, batch in enumerate(dataset.iter(batch_size=batch_size)):
            # Get the batch
            # batch = dataset[i:i+batch_size]
            
            batch["input_ids"] = torch.tensor(batch["input_ids"]).to(device)
            batch["labels"] = torch.tensor(batch["labels"]).to(device)
            batch["attention_mask"] = torch.tensor(batch["attention_mask"]).to(device)
            
            # # Get the english and spanish sentences
            # english = [sentence[0] for sentence in batch]
            # spanish = [sentence[1] for sentence in batch]
            
            # # Tokenize the english and spanish sentences
            # english = model.tokenizer[0](english, padding=True, return_tensors="pt")
            # spanish = model.tokenizer[0](spanish, padding=True, return_tensors="pt")
            
            # Send through model
            output = model(batch)
            
            # Calculate loss
            loss = torch.nn.functional.cross_entropy(output.transpose(-1, -2), batch["labels"])
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print("Epoch:", epoch, "Batch:", i, "Loss:", loss.item())
                
                # Save checkpoint
                if not os.path.exists("checkpoints"):
                    os.mkdir("checkpoints")
                torch.save(model.state_dict(), f"checkpoints/checkpoint-{epoch}-{i}.pt")
            
        print(loss.item())
            
            
            
            
            
            
if __name__ == "__main__":
    main()