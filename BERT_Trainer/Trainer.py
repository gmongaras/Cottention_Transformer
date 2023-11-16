import torch
from torch import nn
import transformers
import datasets
import os
import wandb
from tqdm import tqdm
from contextlib import nullcontext


from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

try:
    from BERT_Trainer.multi_gpu_helpers import is_main_process
    from BERT_Trainer.BertCosAttention import BertCosAttention
except ModuleNotFoundError:
    from multi_gpu_helpers import is_main_process
    from BertCosAttention import BertCosAttention









def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Try the nccl backend
    try:
        dist.init_process_group(
                backend="nccl",
                init_method=dist_url,
                world_size=world_size,
                rank=rank)
    # Use the gloo backend if nccl isn't supported
    except RuntimeError:
        dist.init_process_group(
                backend="gloo",
                init_method=dist_url,
                world_size=world_size,
                rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()














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
            batch_size=256,
            learning_rate=1e-4,
            warmup_steps=10_000,
            num_steps=1_000_000, 
            dev="cpu",
            wandb_name=None,
            log_steps=10,
            use_amp=True,
            attention_type="soft",
            clipping_value=None,
            weight_decay=0.01,
            model_save_path=None,
            num_save_steps=10_000,
        ):
        self.num_steps = num_steps
        self.wandb_name = wandb_name
        self.log_steps = log_steps
        self.use_amp = use_amp
        self.dev = dev
        self.clipping_value = clipping_value
        self.weight_decay = weight_decay
        self.model_save_path = model_save_path.replace(" ", "_") if model_save_path is not None else None
        self.num_save_steps = num_save_steps
        
        
        # Divide the batch size by the number of GPUs
        if dev != "cpu":
            batch_size = batch_size // torch.cuda.device_count()
        else:
            batch_size = batch_size
        self.batch_size = batch_size
        
        
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
        
        
        
        
        # Replace all self attention layers (BertSelfAttention) with the cosine attention layer (BertCosAttention)
        if attention_type == "cos":
            for layer in self.model.bert.encoder.layer:
                old = layer.attention.self
                layer.attention.self = BertCosAttention(self.model.config).to(layer.attention.self.query.weight.device)
                del old
        
        
        
        
        # Put the model on the desired device
        if dev != "cpu":
            # Initialize the environment
            init_distributed()
            
            try:
                local_rank = int(os.environ['LOCAL_RANK'])
            except KeyError:
                local_rank = 0
                print("LOCAL_RANK not found in environment variables. Defaulting to 0.")

            self.model = DDP(self.model.cuda(), device_ids=[local_rank], find_unused_parameters=False)
        else:
            self.model = self.model.cpu()
        
        
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay, eps=1e-7)
        
        # LR Scheduler
        self.scheduler = get_scheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=self.num_steps)
        
        
        # Base model reference for DDP
        if self.dev == "cpu":
            self.model_ref = self.model
        else:
            self.model_ref = self.model.module
        
        
        
        
        # Used to get a random token from the dataset, not including special tokens (0, 101, 102, 103)
        self.tokens = torch.arange(0, self.model_ref.config.vocab_size)
        self.tokens = self.tokens[(self.tokens != 0) & (self.tokens != 101) & (self.tokens != 102) & (self.tokens != 103)]
        
        
        
        
    def augment_data(self, batch):
        # Max lenght of the input
        max_length = max([len(x["input_ids"]) for x in batch])
        
        # 0 if the sentences go together, else 1
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
                
                sentence_pairs_labels.append(1)
                
            # Otherwise, just change the second sentence to type B
            else:
                # Get the index of the [SEP] token
                sep_index = (batch[i]["input_ids"] == 102).nonzero(as_tuple=True)[0][0]
                
                # Change the token type ids to type B
                batch[i]["token_type_ids"][sep_index+1:] = 1
                
                sentence_pairs_labels.append(0)
                
        
        
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
                # If the token is a special token, the label is -100
                if batch[i]["input_ids"][t] == 0 or batch[i]["input_ids"][t] == 102 or batch[i]["input_ids"][t] == 101:
                    labels[t] = -100
                    continue
                
                # Mask 15% of the tokens
                if torch.rand(1) < 0.15:
                    # Update Label
                    labels[t] = batch[i]["input_ids"][t]
                    
                    # 80% of the time, replace with [MASK]
                    if torch.rand(1) < 0.8:
                        # Update token
                        batch[i]["input_ids"][t] = 103
                        
                    # 10% of the time, replace with random token
                    elif torch.rand(1) < 0.5:
                        # Update token. Do not allow special tokens (0, 101, 102, 103)
                        batch[i]["input_ids"][t] = self.tokens[torch.randint(len(self.tokens), (1,))]
                        
                    # 10% of the time, keep the same
                    else:
                        # Label is not -100 as the model still has to predict the token
                        pass
                        
                # Otherwise, keep the same.
                # -100 is the ignore index for the loss function
                else:
                    labels[t] = -100
                    
            # Save the labels
            batch[i]["labels"] = labels
                    
        # Stack the data
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "sentence_pairs_labels": torch.tensor(sentence_pairs_labels, dtype=torch.long)
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
        random_sampler = torch.utils.data.RandomSampler(self.tokenized_dataset, replacement=True, num_samples=self.num_steps*self.batch_size)
        
        # PyTorch data loader
        data_loader = torch.utils.data.DataLoader(
            self.tokenized_dataset, 
            sampler=random_sampler, 
            batch_size=self.batch_size, 
            collate_fn=lambda x: x,
            
            num_workers=10,
            prefetch_factor=10,
            persistent_workers=True,
        )
        
        # Train mode
        self.model.train()
        
        # Initialize wandb run
        if is_main_process():
            wandb.init(
                project="Cos_BERT",
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
        
        loss_fct_MLM = nn.CrossEntropyLoss(ignore_index=-100)
        loss_fct_NSP = nn.CrossEntropyLoss()
        
        # Training loop
        for step, batch in enumerate(tqdm(data_loader)) if is_main_process() else enumerate(data_loader):
            # Set the epoch number for the dataloader to seed the
            # randomization of the sampler
            # if self.dev != "cpu":
            #     data_loader.sampler.set_epoch(step)
            
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
                # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, next_sentence_label=sentence_pairs_labels)
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                
                # Losses for the MLM and NSP
                MLM_loss = loss_fct_MLM(outputs.prediction_logits.view(-1, self.model_ref.config.vocab_size), labels.view(-1).to(outputs.prediction_logits.device))
                NSP_loss = loss_fct_NSP(outputs.seq_relationship_logits, sentence_pairs_labels.to(outputs.seq_relationship_logits.device))
                
                # Total loss
                loss = MLM_loss + NSP_loss
                
            # Backpropagate loss
            if self.use_amp:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Clip gradients
            if self.use_amp:
                grad_scaler.unscale_(self.optimizer)
            if self.clipping_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_value)
            
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
                if is_main_process():
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
            
            
            
            
            if (step+1) % self.num_save_steps == 0:
                self.save_model()
            
            
            
    def save_model(self):
        if is_main_process():
            # Save the model
            self.model_ref.save_pretrained(self.model_save_path)
            self.tokenizer.save_pretrained(self.model_save_path)
            
            # Save the optimizer
            torch.save(self.optimizer.state_dict(), os.path.join(self.model_save_path, "optimizer.pt"))
            
            # Save the scheduler
            torch.save(self.scheduler.state_dict(), os.path.join(self.model_save_path, "scheduler.pt"))
            
            # Save the config
            torch.save({
                "num_steps": self.num_steps,
                "wandb_name": self.wandb_name,
                "log_steps": self.log_steps,
                "use_amp": self.use_amp,
                "dev": self.dev,
                "clipping_value": self.clipping_value,
                "weight_decay": self.weight_decay,
            }, os.path.join(self.model_save_path, "config.pt"))
            
            # Save the tokenizer
            torch.save(self.tokenizer, os.path.join(self.model_save_path, "tokenizer.pt"))
