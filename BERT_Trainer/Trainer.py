import torch
from torch import nn
import transformers
import datasets
import os
import wandb
from tqdm import tqdm
from contextlib import nullcontext
import copy
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
from concurrent.futures import ThreadPoolExecutor


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
            keep_dataset_in_mem=False,
            load_checkpoint=False,
            checkpoint_path=None,
            finetune=False,
            finetune_task=None,
        ):
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.wandb_name = wandb_name
        self.log_steps = log_steps
        self.use_amp = use_amp
        self.dev = dev
        self.clipping_value = clipping_value
        self.weight_decay = weight_decay
        self.model_save_path = model_save_path.replace(" ", "_") if model_save_path is not None else None
        self.num_save_steps = num_save_steps
        self.keep_dataset_in_mem = keep_dataset_in_mem
        self.finetune_ = finetune
        self.finetune_task = finetune_task
        
        
        
        # Must load a checkpoint if finetuning
        if self.finetune_:
            assert load_checkpoint, "Must load a checkpoint if finetuning"
            assert checkpoint_path is not None, "Must provide a checkpoint path if finetuning"


        
        
        
        
        
        # Divide the batch size by the number of GPUs
        if dev != "cpu":
            batch_size = batch_size // int(os.environ['WORLD_SIZE'])
        else:
            batch_size = batch_size
        self.batch_size = batch_size
        
        
        
        # Load in a checkpoint
        if load_checkpoint:
            self.load_checkpoint(checkpoint_path)
            
        # Otherwise initialize from scratch
        else:
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
                    
                    
                    
            # Add attention type to the config
            self.attention_type = attention_type
            
            
            
            
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
            
            # Step starts at 0
            self.step_ckpt = 0
            
            # Wandb id is None
            self.wandb_id = None
            
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
                random_sentence = next(iter(self.random_data_loader))['input_ids'][0]
                
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
        if self.finetune_:
            self.finetune()
        else:
            self.train()
        
        
        
        
        
    def train(self):
        # Cache dirs
        cache_path = "BERT_Trainer/data_cache/dataset_mapped"
        
        # Load in datasets
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.tokenized_dataset = datasets.load_dataset("gmongaras/BERT_Base_Cased_512_Dataset_Mapped", cache_dir=cache_path, num_proc=16, keep_in_memory=self.keep_dataset_in_mem)["train"]
        
        # Load dummy data
        # tokenized_dataset = datasets.load_from_disk("BERT_Trainer/data_cache/dummy_dataset")
        
        # Convert data to torch
        self.tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"])
        
        # PyTorch random sampler
        random_sampler = torch.utils.data.RandomSampler(self.tokenized_dataset, replacement=True, num_samples=(self.num_steps-self.step_ckpt)*self.batch_size)
        
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
        
        
        
        # Data loader for random sentence sampling 
        # Create a sampler for random data fetching
        self.random_data_sampler = torch.utils.data.RandomSampler(self.tokenized_dataset, replacement=True, num_samples=1)
        # Create a data loader for fetching random datapoints
        self.random_data_loader = torch.utils.data.DataLoader(
            self.tokenized_dataset,
            batch_size=1,
            sampler=self.random_data_sampler
        )
        
        
        
        # Train mode
        self.model.train()
        
        # Initialize wandb run
        if is_main_process():
            wandb.init(
                project="Cos_BERT",
                name=self.wandb_name,
                notes=None, # May add notes later
                
                # Resume training if checkpoint exists
                resume="must" if self.wandb_id is not None else None,
                id=self.wandb_id,
            )
            wandb.watch(self.model, log_freq=self.log_steps)
            
            # Save wandb run id
            self.wandb_id = wandb.run.id
        
        # Automatic mixed precision
        if self.use_amp:
            grad_scaler = torch.cuda.amp.GradScaler()
    
        
        batch_MLM_loss = 0
        batch_NSP_loss = 0
        #batch_penalty = 0
        batch_loss = 0
        
        loss_fct_MLM = nn.CrossEntropyLoss(ignore_index=-100)
        loss_fct_NSP = nn.CrossEntropyLoss()
        
        # Training loop
        for step, batch in enumerate(tqdm(data_loader, initial=self.step_ckpt, total=self.num_steps)) if is_main_process() else enumerate(data_loader):
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
                # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True)
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                
                # Losses for the MLM and NSP
                MLM_loss = loss_fct_MLM(outputs.prediction_logits.view(-1, self.model_ref.config.vocab_size), labels.view(-1).to(outputs.prediction_logits.device))
                NSP_loss = loss_fct_NSP(outputs.seq_relationship_logits, sentence_pairs_labels.to(outputs.seq_relationship_logits.device))
                #penalty = 0.01*torch.stack(outputs.attentions).mean()
                
                # Total loss
                loss = MLM_loss + NSP_loss# + penalty
                
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
            self.scheduler.step(step+self.step_ckpt)
            
            # Step the gradient scaler
            if self.use_amp:
                grad_scaler.update()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            
            
            # Update batch losses
            batch_MLM_loss += MLM_loss.item()/self.log_steps
            batch_NSP_loss += NSP_loss.item()/self.log_steps
            #batch_penalty = penalty.item()/self.log_steps
            batch_loss += loss.item()/self.log_steps
            
            
            
            
            # Log wandb
            if (step+self.step_ckpt) % self.log_steps == 0:
                if is_main_process():
                    wandb.log({
                        "MLM loss": batch_MLM_loss,
                        "NSP loss": batch_NSP_loss,
                        #"penalty": batch_penalty,
                        "loss": batch_loss,
                        "lr": self.optimizer.param_groups[0]['lr'],
                    },
                    step=step+self.step_ckpt)
                
                batch_MLM_loss = 0
                batch_NSP_loss = 0
                #batch_penalty = 0
                batch_loss = 0
                
                # if is_main_process():
                #     for i in range(0, len(outputs.attentions)):
                #         wandb.log({f"attn{i}": wandb.Histogram(torch.norm(outputs.attentions[i], dim=-1).cpu().detach().float().numpy())})
            
            # Break if we have reached the max number of steps
            if (step+self.step_ckpt) >= self.num_steps:
                break
            
            
            
            
            if (step+1+self.step_ckpt) % self.num_save_steps == 0:
                self.save_model(step+self.step_ckpt)
                
                
                
            # Clear cache
            # torch.cuda.empty_cache()
            
            
            
            
            
            
            
            
            
    def pad_batch(self, batch):
        """
        Pad the batch of sequences using the tokenizer.
        Args:
            batch (DataFrame): The batch to pad.
            tokenizer: The tokenizer to use for padding.
        Returns:
            DataFrame: The padded batch.
        """
        # Convert DataFrame to list of dictionaries and then to a dictionary of lists
        batch_dict = batch.to_dict(orient='list')

        # Get the maximum sequence length in the batch
        max_length = max(len(ids) for ids in batch_dict["input_ids"])
        
        # Convert lists to tensors and pad them
        batch_dict["input_ids"] = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids, dtype=torch.long) for ids in batch_dict["input_ids"]], 
                                                                batch_first=True, padding_value=self.tokenizer.pad_token_id)
        batch_dict["attention_mask"] = torch.nn.utils.rnn.pad_sequence([torch.tensor(mask, dtype=torch.long) for mask in batch_dict["attention_mask"]], 
                                                                    batch_first=True, padding_value=0)
        batch_dict["token_type_ids"] = torch.nn.utils.rnn.pad_sequence([torch.tensor(tti, dtype=torch.long) for tti in batch_dict["token_type_ids"]], 
                                                                    batch_first=True, padding_value=1)

        # Update token type ids for [SEP] token
        for i, input_ids in enumerate(batch_dict["input_ids"]):
            sep_index = (input_ids == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[0][0]
            batch_dict["token_type_ids"][i, sep_index+1:] = 1

        # Convert labels to a tensor
        batch_dict["labels"] = torch.tensor(batch_dict["label"], dtype=torch.float)
        # Assuming dataset_name is constant for a batch, convert to a tensor
        batch_dict["dataset_name"] = torch.tensor(batch_dict["dataset_name"][0], dtype=torch.long)

        return {
            "input_ids": batch_dict["input_ids"],
            "attention_mask": batch_dict["attention_mask"],
            "token_type_ids": batch_dict["token_type_ids"],
            "labels": batch_dict["labels"],
            "dataset_name": batch_dict["dataset_name"],
        }
    
    
    def prepare_batches(self, datasets):
        """
        Process a list of datasets (each a separate group), group by batch size, pad each batch, and convert 'dataset_name' to numerical.
        Args:
            datasets (list of DataFrame): List of DataFrames, each DataFrame is a group.
            tokenizer: The tokenizer to use for padding.
            batch_size (int): The size of each batch.
        Returns:
            DataFrame: All groups merged and processed into a single DataFrame.
        """
        processed_groups = []
        dataset_name_to_num = {dataset['dataset_name'].iloc[0]: i for i, dataset in enumerate(datasets)}
        batch_indices = np.arange(self.batch_size)

        for dataset in datasets:
            # Convert 'dataset_name' to numerical
            dataset['dataset_name'] = dataset_name_to_num[dataset['dataset_name'].iloc[0]]

            # Calculate number of batches
            num_batches = int(np.ceil(len(dataset) / self.batch_size))

            # Process each batch
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size

                # Select and pad the batch
                batch = dataset.iloc[start_idx:end_idx]
                padded_batch = self.pad_batch(batch)

                # Append the padded batch
                processed_groups.append(padded_batch)

        # Save the dataset mapping information
        self.dataset_name_to_num = dataset_name_to_num

        return processed_groups
            
            
    def finetune(self):
        # Cache dirs
        cache_path = "BERT_Trainer/data_cache/dataset_ft_mapped"
        
        # Load in datasets
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.tokenized_dataset = datasets.load_dataset("gmongaras/BERT_Base_Cased_512_GLUE_Mapped", cache_dir=cache_path, num_proc=16, keep_in_memory=self.keep_dataset_in_mem)
        
        # Subset the dataset and shuffle it
        for dataset_name in self.tokenized_dataset.keys():
            self.tokenized_dataset[dataset_name] = self.tokenized_dataset[dataset_name].filter(lambda x: x["dataset_name"] == self.finetune_task).shuffle()
        
        # Convert data to torch
        self.tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
        
        # Get train and validation datasets
        train_dataset = self.tokenized_dataset["train"]
        val_dataset = self.tokenized_dataset["validation"]

        # Convert to pandas
        train_dataset = train_dataset.to_pandas()
        val_dataset = val_dataset.to_pandas()
        
        # Group the huggingface datasets by "dataset_name"
        train_dataset = train_dataset.groupby("dataset_name")
        val_dataset = val_dataset.groupby("dataset_name")
        train_dataset = [train_dataset.get_group(x) for x in train_dataset.groups]
        val_dataset = [val_dataset.get_group(x) for x in val_dataset.groups]
        
        train_dataset = self.prepare_batches(train_dataset)
        val_dataset = self.prepare_batches(val_dataset)
        
        
        
        
        # Train mode
        self.model.train()
        
        # Initialize wandb run
        if is_main_process():
            wandb.init(
                project="Cos_BERT",
                name=self.wandb_name,
                notes=None, # May add notes later
            )
            wandb.watch(self.model, log_freq=1)
        
        # Save wandb run id
        self.wandb_id = wandb.run.id
        
        # Automatic mixed precision
        if self.use_amp:
            grad_scaler = torch.cuda.amp.GradScaler()
            
            
            
        # Iterate for three epochs
        for epoch in range(3):
            batch_loss = 0
            batch_acc = 0
            
            # Shuffle dataset
            np.random.shuffle(train_dataset)
            
            # Training loop
            for step, batch in enumerate(tqdm(train_dataset)) if is_main_process() else enumerate(train_dataset):
                # Get input and labels
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                token_type_ids = batch["token_type_ids"].to(self.model.device)
                labels = batch["labels"].to(self.model.device)
                dataset_name = batch["dataset_name"].cpu().item()
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                    # Get model predictions
                    # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, next_sentence_label=sentence_pairs_labels)
                    # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True)
                    outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
                    
                    # Output hidden states
                    outputs = outputs.hidden_states[-1][:, 0]
                    
                    # Linear layer
                    outputs = self.model_ref.head(outputs)
                    
                    # MSE loss if stsb
                    if self.num_to_dataset_name[dataset_name] == "stsb":
                        loss = torch.nn.MSELoss()(outputs, labels.to(outputs.device))
                    # Cross entropy loss otherwise
                    else:
                        loss = torch.nn.CrossEntropyLoss()(outputs, labels.long().to(outputs.device))
                    
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
                
                # # Update scheduler
                # self.scheduler.step(step+self.step_ckpt)
                
                # Step the gradient scaler
                if self.use_amp:
                    grad_scaler.update()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                
                
                # Update batch losses
                batch_loss += loss.item()/len(train_dataset)
                
                # Update batch accuracy
                if self.num_to_dataset_name[dataset_name] == "stsb":
                    batch_acc += torch.abs(outputs - labels.to(outputs.device)).sum().item()/len(train_dataset)
                else:
                    batch_acc += (outputs.argmax(dim=-1) == labels.long().to(outputs.device)).sum().item()/len(train_dataset)
                
                
            
            
            
            print("Validating...")
            self.model.eval()
            val_batch_loss = 0
            val_acc = 0
            # Validation loop
            for step, batch in enumerate(tqdm(val_dataset)) if is_main_process() else enumerate(val_dataset):
                # Get input and labels
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)
                token_type_ids = batch["token_type_ids"].to(self.model.device)
                labels = batch["labels"].to(self.model.device)
                dataset_name = batch["dataset_name"].cpu().item()
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                    # Get model predictions
                    # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, next_sentence_label=sentence_pairs_labels)
                    # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True)
                    outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
                    
                    # Output hidden states
                    outputs = outputs.hidden_states[-1][:, 0]
                    
                    # Linear layer
                    outputs = self.model_ref.head(outputs)
                    
                    # MSE loss if stsb
                    if self.num_to_dataset_name[dataset_name] == "stsb":
                        loss = torch.nn.MSELoss()(outputs, labels.to(outputs.device))
                    # Cross entropy loss otherwise
                    else:
                        loss = torch.nn.CrossEntropyLoss()(outputs, labels.long().to(outputs.device))
                    
                # Update batch losses
                val_batch_loss += loss.item()/len(val_dataset)
                
                # Update batch accuracy
                if self.num_to_dataset_name[dataset_name] == "stsb":
                    val_acc += torch.abs(outputs - labels.to(outputs.device)).sum().item()/len(val_dataset)
                else:
                    val_acc += (outputs.argmax(dim=-1) == labels.long().to(outputs.device)).sum().item()/len(val_dataset)
                
            self.model.train()
                
                
                
                
            # Log wandb
            if is_main_process():
                wandb_args = {
                    "loss_finetune": batch_loss,
                    "acc_finetune": batch_acc,
                    "val_loss_finetune": val_batch_loss,
                    "val_acc_finetune": val_acc,
                }
                
                wandb.log(wandb_args)
            
            batch_loss = 0
            
            
            
            # if (step+1+self.step_ckpt) % self.num_save_steps == 0:
            #     self.save_model(step+self.step_ckpt)
                
                
                
            # Clear cache
            # torch.cuda.empty_cache()
            
            
            
    def save_model(self, step):
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
                "learning_rate": self.learning_rate,
                "warmup_steps": self.warmup_steps,
                "num_steps": self.num_steps,
                "wandb_name": self.wandb_name,
                "log_steps": self.log_steps,
                "use_amp": self.use_amp,
                "dev": self.dev,
                "clipping_value": self.clipping_value,
                "weight_decay": self.weight_decay,
                "attention_type": self.attention_type,
                "step_ckpt": step,
                "wandb_id": self.wandb_id,
            }, os.path.join(self.model_save_path, "config.pt"))
            
            # Save the tokenizer
            torch.save(self.tokenizer, os.path.join(self.model_save_path, "tokenizer.pt"))
            
            
            
    def load_checkpoint(self, checkpoint_path):
        # Load the model
        self.model = transformers.BertForPreTraining.from_pretrained(checkpoint_path.replace(" ", "_"))
        
        # Load the config
        config = torch.load(os.path.join(checkpoint_path, "config.pt"))
        self.learning_rate = config["learning_rate"]
        self.warmup_steps = config["warmup_steps"]
        self.num_steps = config["num_steps"]
        if not self.finetune_:
            self.wandb_name = config["wandb_name"]
        self.log_steps = config["log_steps"]
        self.use_amp = config["use_amp"]
        self.dev = config["dev"]
        self.clipping_value = config["clipping_value"]
        self.weight_decay = config["weight_decay"]
        self.step_ckpt = config["step_ckpt"]
        self.attention_type = config["attention_type"]
        self.wandb_id = config["wandb_id"]
        
        # Replace all self attention layers (BertSelfAttention) with the cosine attention layer (BertCosAttention)
        if self.attention_type == "cos":
            for layer in self.model.bert.encoder.layer:
                old = layer.attention.self
                layer.attention.self = BertCosAttention(self.model.config).to(layer.attention.self.query.weight.device)
                
                # Copy weights
                layer.attention.self.query.weight.data = old.query.weight.data
                layer.attention.self.query.bias.data = old.query.bias.data
                layer.attention.self.key.weight.data = old.key.weight.data
                layer.attention.self.key.bias.data = old.key.bias.data
                layer.attention.self.value.weight.data = old.value.weight.data
                layer.attention.self.value.bias.data = old.value.bias.data
                
                del old
                
            # Load extra params if needed
            self.model.load_state_dict(torch.load(checkpoint_path.replace(" ", "_") + "/pytorch_model.bin", map_location=self.model.bert.encoder.layer[0].attention.self.query.weight.device))
            
            # Clear cache
            torch.cuda.empty_cache()
        
        # Load the tokenizer
        self.tokenizer = torch.load(os.path.join(checkpoint_path, "tokenizer.pt"))
        
        
        # Extra params for finetuning
        if self.finetune_:
            # Freeze the MLM and NSP heads
            for param in self.model.cls.parameters():
                param.requires_grad = False
        
        
        # Put the model on the desired device
        if self.dev != "cpu":
            if self.finetune_:
                self.model = self.model.cuda()
                
                self.model_ref = self.model
            else:
                # Initialize the environment
                init_distributed()
                
                try:
                    local_rank = int(os.environ['LOCAL_RANK'])
                except KeyError:
                    local_rank = 0
                    print("LOCAL_RANK not found in environment variables. Defaulting to 0.")

                self.model = DDP(self.model.cuda(), device_ids=[local_rank], find_unused_parameters=False)
                self.model_ref = self.model.module
        else:
            self.model = self.model.cpu()
            
            self.model_ref = self.model
        
        
        # New optimizer if finetuning
        if self.finetune_:
            # Dataset to number mapping
            self.dataset_name_to_num = {
                "cola": 0,
                "mnli": 1,
                "mrpc": 2,
                "qnli": 3,
                "qqp": 4,
                "rte": 5,
                "sst2": 6,
                "stsb": 7,
                "wnli": 8,
            }
            self.num_to_dataset_name = {v: k for k, v in self.dataset_name_to_num.items()}
            
            # Assert that the task is valid
            assert self.finetune_task in self.dataset_name_to_num, f"Invalid finetune task {self.finetune_task}"
            
            # Heads for finetuning tasks
            if self.finetune_task == "cola":
                self.model.head = nn.Linear(self.model.config.hidden_size, 2, device=self.model.device)
            elif self.finetune_task == "mnli":
                self.model.head = nn.Linear(self.model.config.hidden_size, 3, device=self.model.device)
            elif self.finetune_task == "mrpc":
                self.model.head = nn.Linear(self.model.config.hidden_size, 2, device=self.model.device)
            elif self.finetune_task == "qnli":
                self.model.head = nn.Linear(self.model.config.hidden_size, 2, device=self.model.device)
            elif self.finetune_task == "qqp":
                self.model.head = nn.Linear(self.model.config.hidden_size, 2, device=self.model.device)
            elif self.finetune_task == "rte":
                self.model.head = nn.Linear(self.model.config.hidden_size, 2, device=self.model.device)
            elif self.finetune_task == "sst2":
                self.model.head = nn.Linear(self.model.config.hidden_size, 2, device=self.model.device)
            elif self.finetune_task == "stsb":
                self.model.head = nn.Linear(self.model.config.hidden_size, 1, device=self.model.device)
            elif self.finetune_task == "wnli":
                self.model.head = nn.Linear(self.model.config.hidden_size, 2, device=self.model.device)
            
            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay, eps=1e-7)
            
        # Load checkpoint for optimizer if not finetuning
        else:
            # Load the optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay, eps=1e-7)
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), map_location=self.model_ref.bert.encoder.layer[0].attention.self.query.weight.device))
            
            # Load the scheduler
            self.scheduler = get_scheduler(self.optimizer, warmup_steps=self.warmup_steps, total_steps=self.num_steps)
            self.scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scheduler.pt"), map_location=self.model_ref.bert.encoder.layer[0].attention.self.query.weight.device))
