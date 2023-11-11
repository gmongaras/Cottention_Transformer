import torch
from torch import nn
import transformers
import datasets
import os




def get_scheduler(optimizer, warmup_steps, total_steps):
    # Define the lambda function for the learning rate schedule
    lr_lambda = lambda step: (
        1.0 if step < warmup_steps 
        else (1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    )

    # Create the scheduler
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)





class Trainer():
    def __init__(self, num_steps=1_000_000):
        self.num_steps = num_steps
        
        
        # Tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False, cache_dir="BERT_Trainer/BERT_Model")
        # BERT Model. We are training it from scratch
        self.model = transformers.BertForPreTraining(config=transformers.BertConfig.from_dict({
            "architectures": [
                "BertForMaskedLM"
            ],
            "attention_probs_dropout_prob": 0.1,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": 28996
        }))
        
        
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
        
        # LR Scheduler
        self.scheduler = get_scheduler(self.optimizer, warmup_steps=10_000, total_steps=self.num_steps)
        
        
    def __call__(self):
        # Cache dirs
        cache_path = "BERT_Trainer/data_cache/dataset"
        
        # Load in datasets
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        wiki_tokenized_dataset = datasets.load_dataset("gmongaras/BERT_Base_Cased_512_Dataset", cache_dir=cache_path)["train"]
        
        # Load dummy data
        # wiki_tokenized_dataset = datasets.load_from_disk("BERT_Trainer/data_cache/dummy_dataset")
        
        # Convert data to torch
        wiki_tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"])
        
        # Batch dataset (batch size 8)
        wiki_tokenized_dataset = wiki_tokenized_dataset.shard(index=0, num_shards=8)
        
        # PyTorch random sampler
        random_sampler = torch.utils.data.RandomSampler(wiki_tokenized_dataset)
        
        # PyTorch data loader
        data_loader = torch.utils.data.DataLoader(wiki_tokenized_dataset, sampler=random_sampler, batch_size=8)
        
        
        
        
        # Training loop
        for step, batch in enumerate(data_loader):
            # Get input and output
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            
            # Augment input
            
            # Get loss
            outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).prediction_logits
            loss = outputs.loss
            
            # Backpropagate loss
            loss.backward()
            
            # Take optimizer step
            self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Print loss
            if step % 100 == 0:
                print(f"Step: {step} | Loss: {loss.item()}")
            
            # Break if we have reached the max number of steps
            if step >= self.num_steps:
                break