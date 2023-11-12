import torch
from torch import nn
import transformers
import datasets
import os
import wandb
from tqdm import tqdm
from contextlib import nullcontext




def get_scheduler(optimizer, warmup_steps, total_steps):
    # Define the lambda function for the learning rate schedule
    # this value 
    lr_lambda = lambda step: (
        step/warmup_steps if step < warmup_steps
        else (1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    )

    # Create the scheduler
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)





class Trainer():
    def __init__(self, 
            batch_size=32,
            learning_rate=1e-4,
            warmup_steps=10_000,
            num_steps=1_000_000, 
            device=torch.device("cpu"),
            wandb_name=None,
            log_steps=10,
            use_amp=True,
        ):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.device = device
        self.wandb_name = wandb_name
        self.log_steps = log_steps
        self.use_amp = use_amp
        
        
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
        })).to(self.device)
        
        
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-7)
        
        # LR Scheduler
        self.scheduler = get_scheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=self.num_steps)
        
        
        
        
    def augment_data(self, batch):
        # Max lenght of the input
        max_length = max([len(x["input_ids"]) for x in batch])
        
        # 1 if the sentences go together, else 1
        sentence_pairs_labels = []
        
        
        for i in range(len(batch)):
            ### First, augment sentences with random sentences
            
            
            # Augment 50% of the sentences
            if torch.rand(1) < 0.5:
                # Get a random sentence from the dataset
                random_sentence = self.tokenized_dataset[torch.randint(len(self.tokenized_dataset), (1,))]["input_ids"][0]
                
                # Randomly get the first or second sentence in the pair
                idxs = (random_sentence == 102).nonzero(as_tuple=True)[0]
                if torch.rand(1) < 0.5:
                    random_sentence = random_sentence[1:idxs[0]]
                else:
                    random_sentence = random_sentence[idxs[0]+1:]
                
                # Get the index of the [SEP] token
                sep_index = (batch[i]["input_ids"] == 102).nonzero(as_tuple=True)[0][0]
                
                # Insert the sentence as a new sentence with the [SEP] token and the second
                # sentence as the "B" sentence
                batch[i]["input_ids"] = torch.cat([batch[i]["input_ids"][:sep_index+1], random_sentence])
                batch[i]["attention_mask"] = torch.cat([batch[i]["attention_mask"][:sep_index+1], torch.ones_like(random_sentence)])
                batch[i]["token_type_ids"] = torch.cat([batch[i]["token_type_ids"][:sep_index+1], torch.ones_like(random_sentence)])
                
                sentence_pairs_labels.append(0)
                
            # Otherwise, just change the second sentence to type B
            else:
                # Get the index of the [SEP] token
                sep_index = (batch[i]["input_ids"] == 102).nonzero(as_tuple=True)[0][0]
                
                # Change the token type ids to type B
                batch[i]["token_type_ids"][sep_index+1:] = 1
                
                sentence_pairs_labels.append(1)
                
        
        
            ### Trim the input to max length
            batch[i]["input_ids"] = batch[i]["input_ids"][:max_length]
            batch[i]["attention_mask"] = batch[i]["attention_mask"][:max_length]
            batch[i]["token_type_ids"] = batch[i]["token_type_ids"][:max_length]
        
            ### Pad the input to max length
            batch[i]["input_ids"] = torch.cat([batch[i]["input_ids"], torch.zeros(max_length - len(batch[i]["input_ids"]), dtype=torch.long)])
            batch[i]["attention_mask"] = torch.cat([batch[i]["attention_mask"], torch.zeros(max_length - len(batch[i]["attention_mask"]), dtype=torch.long)])
            batch[i]["token_type_ids"] = torch.cat([batch[i]["token_type_ids"], torch.ones(max_length - len(batch[i]["token_type_ids"]), dtype=torch.long)])
            
            
            
            ### Mask 15% of the tokens
            
            # Labels for the MLM
            labels = batch[i]["input_ids"].clone()
            
            for t in range(len(batch[i]["input_ids"])):
                # If the token is a padding token, the label is -100
                if batch[i]["input_ids"][t] == 0:
                    labels[t] = -100
                    continue
                
                # Mask 15% of the tokens
                if torch.rand(1) < 0.15:
                    # 80% of the time, replace with [MASK]
                    if torch.rand(1) < 0.8:
                        # Update Label
                        labels[t] = batch[i]["input_ids"][t]
                        
                        # Update token
                        batch[i]["input_ids"][t] = 103
                        
                    # 10% of the time, replace with random token
                    elif torch.rand(1) < 0.5:
                        # Update Label
                        labels[t] = batch[i]["input_ids"][t]
                        
                        # Update token
                        batch[i]["input_ids"][t] = torch.randint(self.tokenizer.vocab_size, (1,))
                        
                    # 10% of the time, keep the same
                    else:
                        # Label is not -100
                        labels[t] = batch[i]["input_ids"][t]
                        
                # Otherwise, keep the same.
                # -100 is the ignore index for the loss function
                else:
                    labels[t] = -100
                    
            # Save the labels
            batch[i]["labels"] = labels
                    
        # Stack the data
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]).to(self.device),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]).to(self.device),
            "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]).to(self.device),
            "labels": torch.stack([x["labels"] for x in batch]).to(self.device),
            "sentence_pairs_labels": torch.tensor(sentence_pairs_labels, dtype=torch.long).to(self.device)
        }
        
            
        
        
        
        
        
        
    def __call__(self):
        # Cache dirs
        cache_path = "BERT_Trainer/data_cache/dataset_mapped"
        
        # Load in datasets
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.tokenized_dataset = datasets.load_dataset("gmongaras/BERT_Base_Cased_512_Dataset_Mapped", cache_dir=cache_path, num_proc=16)["train"]
        
        # Load dummy data
        # tokenized_dataset = datasets.load_from_disk("BERT_Trainer/data_cache/dummy_dataset")
        
        # Convert data to torch
        self.tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"])
        
        # PyTorch random sampler
        random_sampler = torch.utils.data.RandomSampler(self.tokenized_dataset, replacement=True)
        
        # PyTorch data loader
        data_loader = torch.utils.data.DataLoader(self.tokenized_dataset, sampler=random_sampler, batch_size=self.batch_size, collate_fn=lambda x: x)
        
        # Train mode
        self.model.train()
        
        # Initialize wandb run
        wandb.init(
            project="Cottention",
            name=self.wandb_name,
            notes=None, # May add notes later
            
            # # Resume training if checkpoint exists
            # resume="must" if wandb_id is not None else None,
            # id=wandb_id,
        )
        wandb.watch(self.model, log_freq=self.log_steps)
        
        # Automatic mixed precision
        if self.use_amp:
            grad_scaler = torch.cuda.amp.GradScaler()
    
        
        batch_MLM_loss = 0
        batch_NSP_loss = 0
        batch_loss = 0
        
        # Training loop
        for step, batch in enumerate(tqdm(data_loader)):
            # Set the epoch number for the dataloader to seed the
            # randomization of the sampler
            # if self.dev != "cpu":
            #     self.train_dataloader.sampler.set_epoch(step)
            # data_loader.sampler.set_epoch(step)
            
            # Augment input
            batch = self.augment_data(batch)
            
            # Get input and labels
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["labels"]
            sentence_pairs_labels = batch["sentence_pairs_labels"]
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                # Get model predictions
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                
                # Losses for the MLM and NSP
                loss_fct_MLM = nn.CrossEntropyLoss(ignore_index=-100)
                loss_fct_NSP = nn.CrossEntropyLoss()
                MLM_loss = loss_fct_MLM(outputs.prediction_logits.view(-1, self.model.config.vocab_size), labels.view(-1))
                NSP_loss = loss_fct_NSP(outputs.seq_relationship_logits, sentence_pairs_labels)
                
                # Total loss
                loss = MLM_loss + NSP_loss
                
            # Backpropagate loss
            if self.use_amp:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Take optimizer step
            if self.use_amp:
                grad_scaler.step(self.optimizer)
            else:
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step(step)
            
            # Step the gradient scaler
            if self.use_amp:
                grad_scaler.update()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            
            
            # Update batch losses
            batch_MLM_loss += MLM_loss.item()/self.log_steps
            batch_NSP_loss += NSP_loss.item()/self.log_steps
            batch_loss += loss.item()/self.log_steps
            
            
            
            
            # Log wandb
            if step % self.log_steps == 0:
                wandb.log({
                    "MLM loss": batch_MLM_loss,
                    "NSP loss": batch_NSP_loss,
                    "loss": batch_loss,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "step": step,
                })
                
                batch_MLM_loss = 0
                batch_NSP_loss = 0
                batch_loss = 0
            
            # Break if we have reached the max number of steps
            if step >= self.num_steps:
                break
