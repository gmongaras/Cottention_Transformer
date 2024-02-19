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
    from GPT_Trainer.multi_gpu_helpers import is_main_process
    from GPT_Trainer.GPTCosAttention import GPTCosAttention
except ModuleNotFoundError:
    from multi_gpu_helpers import is_main_process
    from GPTCosAttention import GPTCosAttention









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
            self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", use_fast=False, cache_dir="GPT_Trainer/gpt-j-6B")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token = torch.tensor([self.tokenizer.pad_token_id])
            # GPT-J Model. We are training it from scratch
            self.model = transformers.GPTJForCausalLM(config=transformers.GPTJConfig.from_dict({
                "activation_function": "gelu_new",
                "architectures": [
                    "GPTJForCausalLM"
                ],
                "attn_pdrop": 0.0,
                "bos_token_id": 50256,
                "embd_pdrop": 0.0,
                "eos_token_id": 50256,
                "gradient_checkpointing": False,
                "initializer_range": 0.02,
                "layer_norm_epsilon": 1e-05,
                "model_type": "gptj",
                "n_embd": 2048, # 4096,
                "n_head": 16,
                "n_inner": None,
                "n_layer": 20, #28,
                "n_positions": 2048,
                "resid_pdrop": 0.0,
                "rotary": True,
                "rotary_dim": 64,
                "scale_attn_weights": True,
                "summary_activation": None,
                "summary_first_dropout": 0.1,
                "summary_proj_to_labels": True,
                "summary_type": "cls_index",
                "summary_use_proj": True,
                "task_specific_params": {
                    "text-generation": {
                    "do_sample": True,
                    "max_length": 50,
                    "temperature": 1.0
                    }
                },
                "tie_word_embeddings": False,
                "tokenizer_class": "GPT2Tokenizer",
                "transformers_version": "4.18.0.dev0",
                "use_cache": True,
                "vocab_size": 50400
            }))
            
            
            # Replace all self attention layers (BertSelfAttention) with the cosine attention layer (GPTCosAttention)
            if attention_type == "cos":
                for layer in self.model.transformer.h:
                    old = layer.attn
                    layer.attn = GPTCosAttention(self.model.config).to(layer.attn.q_proj.weight.device)
                    del old
                    
                    
                    
            # Add attention type to the config
            self.attention_type = attention_type
            
            
            
            
            # Put the model on the desired device
            if dev != "cpu":
                # Initialize the environment
                if not dist.is_initialized():
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
        
            
        
    def prepare_data(self, batch):
        # Max lenght of the input
        max_length = max([len(x["input_ids"]) for x in batch]) + 1 # +1 for the extra pad token
        
        
        for i in range(len(batch)):
            ### Trim the input to max length
            batch[i]["input_ids"] = batch[i]["input_ids"][:self.tokenizer.model_max_length]
            batch[i]["attention_mask"] = batch[i]["attention_mask"][:self.tokenizer.model_max_length]
            batch[i]["token_type_ids"] = batch[i]["token_type_ids"][:self.tokenizer.model_max_length]
            
            ### Add a pad token to the end without mask to make the model stop itself
            batch[i]["input_ids"] = torch.cat([batch[i]["input_ids"], self.pad_token])
            batch[i]["attention_mask"] = torch.cat([batch[i]["attention_mask"], torch.tensor([1])])
            batch[i]["token_type_ids"] = torch.cat([batch[i]["token_type_ids"], torch.tensor([0])])
        
            ### Pad the input to max length
            batch[i]["input_ids"] = torch.cat([batch[i]["input_ids"], torch.zeros(max_length - len(batch[i]["input_ids"]), dtype=torch.long)])
            batch[i]["attention_mask"] = torch.cat([batch[i]["attention_mask"], torch.zeros(max_length - len(batch[i]["attention_mask"]), dtype=torch.long)]).bool()
            batch[i]["token_type_ids"] = torch.cat([batch[i]["token_type_ids"], torch.ones(max_length - len(batch[i]["token_type_ids"]), dtype=torch.long)])
            
            ### Labels are input ids shifted by one. Remove the last token from the others to match the labels
            batch[i]["labels"] = batch[i]["input_ids"].clone()[1:]
            batch[i]["input_ids"] = batch[i]["input_ids"][:-1]
            batch[i]["attention_mask"] = batch[i]["attention_mask"][:-1]
            batch[i]["token_type_ids"] = batch[i]["token_type_ids"][:-1]
                    
        # Stack the data
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
        }
        
        
        
        
    def __call__(self):
        if self.finetune_:
            self.finetune()
        else:
            self.train_model()
            
            
            
            
    def train_model(self):
        self.train_model_("gmongaras/BERT_Base_Cased_512_Dataset_Mapped", self.num_steps, self.step_ckpt)
        
        
        
        
        
    def train_model_(self, dataset, num_steps, step_shift):
        # Cache dirs
        cache_path = "BERT_Trainer/data_cache/dataset_mapped"
        
        # Load in datasets
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.tokenized_dataset = datasets.load_dataset(dataset, cache_dir=cache_path, num_proc=16, keep_in_memory=self.keep_dataset_in_mem)["train"]
        
        # Load dummy data
        # tokenized_dataset = datasets.load_from_disk("BERT_Trainer/data_cache/dummy_dataset")
        
        # Convert data to torch
        self.tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids"])
        
        # PyTorch random sampler
        random_sampler = torch.utils.data.RandomSampler(self.tokenized_dataset, replacement=True, num_samples=(num_steps-step_shift)*self.batch_size)
        
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
                project="Cos_GPT",
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
    
        
        batch_loss = 0
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training loop
        for step, batch in enumerate(tqdm(data_loader, initial=step_shift, total=num_steps)) if is_main_process() else enumerate(data_loader):
            step += step_shift
            
            # Set the epoch number for the dataloader to seed the
            # randomization of the sampler
            # if self.dev != "cpu":
            #     data_loader.sampler.set_epoch(step)
            
            # Augment input
            batch = self.prepare_data(batch)
            
            # Get input and labels
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["labels"]
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                # Get model predictions
                # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, next_sentence_label=sentence_pairs_labels)
                # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True)
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                
                # Loss
                loss = loss_fct(outputs.logits.view(-1, self.model_ref.config.vocab_size), labels.view(-1).to(outputs.logits.device))
                
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
            
            
            
            # Update batch loss
            batch_loss += loss.item()/self.log_steps
            
            
            
            
            # Log wandb
            if (step) % self.log_steps == 0:
                if is_main_process():
                    wandb.log({
                        "loss": batch_loss,
                        "perplexity": torch.exp(torch.tensor(batch_loss)),
                        "lr": self.optimizer.param_groups[0]['lr'],
                    },
                    step=step)
                
                batch_loss = 0
            
            # Break if we have reached the max number of steps
            if (step) >= self.num_steps:
                break
            
            
            
            
            if step % self.num_save_steps == 0:
                self.save_model(step)
                
                
                
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
        batch_indices = np.arange(self.batch_size)

        for dataset in datasets:
            # Shuffle the dataset
            dataset = dataset.sample(frac=1).reset_index(drop=True, inplace=False)
            
            # Convert 'dataset_name' to numerical
            dataset['dataset_name'] = self.dataset_name_to_num[dataset['dataset_name'].iloc[0]]

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

        return processed_groups
            
            
    def finetune(self):
        # Cache dirs
        cache_path = "GPT_Trainer/data_cache/dataset_ft_mapped"
        
        # Load in datasets
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.tokenized_dataset = datasets.load_dataset("gmongaras/BERT_Base_Cased_512_GLUE_Mapped", cache_dir=cache_path, num_proc=16, keep_in_memory=self.keep_dataset_in_mem)
        
        # Subset the dataset and shuffle it
        for dataset_name in self.tokenized_dataset.keys():
            if self.finetune_task == "mnli" and dataset_name != "train":
                self.tokenized_dataset[dataset_name] = {
                    "matched": self.tokenized_dataset[dataset_name].filter(lambda x: x["dataset_name"] == "mnli_matched_validation"),
                    "mismatched": self.tokenized_dataset[dataset_name].filter(lambda x: x["dataset_name"] == "mnli_mismatched_validation"),
                }
            else:
                self.tokenized_dataset[dataset_name] = self.tokenized_dataset[dataset_name].filter(lambda x: x["dataset_name"] == self.finetune_task).shuffle()
        
        # Convert data to torch
        if self.finetune_task != "mnli":
            self.tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
        else:
            self.tokenized_dataset["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
            self.tokenized_dataset["validation"]["matched"].set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
            self.tokenized_dataset["validation"]["mismatched"].set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
        
        # Get train and validation datasets
        train_dataset_ = [self.tokenized_dataset["train"].to_pandas()]
        if self.finetune_task == "mnli":
            val_dataset_ = {
                "matched": [self.tokenized_dataset["validation"]["matched"].to_pandas()],
                "mismatched": [self.tokenized_dataset["validation"]["mismatched"].to_pandas()],
            }
        else:
            val_dataset_ = [self.tokenized_dataset["validation"].to_pandas()]
        
        # # Group the huggingface datasets by "dataset_name"
        # train_dataset = train_dataset.groupby("dataset_name")
        # val_dataset = val_dataset.groupby("dataset_name")
        # train_dataset = [train_dataset.get_group(x) for x in train_dataset.groups]
        # val_dataset = [val_dataset.get_group(x) for x in val_dataset.groups]
        
        
        
        
        # Train mode
        self.model.train()
        
        # Initialize wandb run
        if is_main_process():
            wandb.init(
                project="Cos_GPT_Finetune",
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
        for epoch in range(4):
            batch_loss = 0
            batch_acc = 0
            
            # Batch the dataset
            train_dataset = self.prepare_batches(train_dataset_)
            if self.finetune_task == "mnli":
                val_dataset = {
                    "matched": self.prepare_batches(val_dataset_["matched"]),
                    "mismatched": self.prepare_batches(val_dataset_["mismatched"]),
                }
            else:
                val_dataset = self.prepare_batches(val_dataset_)
            
            # Stored outputs and labels
            outputs_ = []
            labels_ = []
            
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
                    outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
                    
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
                if self.num_to_dataset_name[dataset_name] == "qqp" or self.num_to_dataset_name[dataset_name] == "mrpc":
                    # F1 score
                    TP = ((outputs.argmax(dim=-1) == labels.long().to(outputs.device)) & (labels == 1)).sum().item()
                    FP = ((outputs.argmax(dim=-1) != labels.long().to(outputs.device)) & (outputs.argmax(dim=-1) == 1)).sum().item()
                    FN = ((outputs.argmax(dim=-1) != labels.long().to(outputs.device)) & (outputs.argmax(dim=-1) == 0)).sum().item()
                    
                    batch_acc += TP/(TP + 0.5*(FP+FN))/len(train_dataset) if TP + 0.5*(FP+FN) != 0 else 0
                else:
                    outputs_ += list(outputs.squeeze(-1).float().cpu().detach().numpy())
                    labels_ += list(labels.float().cpu().detach().numpy())
                
            # Calculate the stsb Pearson correlation
            if self.num_to_dataset_name[dataset_name] == "stsb":
                outputs_ = np.array(outputs_)
                labels_ = np.array(labels_)
                # batch_acc = 1 - (
                #         (6*((outputs_ - labels_)**2).sum())
                #         / (outputs_.shape[0] * (outputs_.shape[0]**2 - 1))
                #     )
                batch_acc = abs(np.corrcoef(outputs_, labels_)[0, 1])
            elif self.num_to_dataset_name[dataset_name] != "qqp" and self.num_to_dataset_name[dataset_name] != "mrpc":
                batch_acc = (np.stack(outputs_).argmax(-1)==np.stack(labels_)).astype(float).mean()
            
            print("Validating...")
            self.model.eval()
            def val_loop(dataset):
                val_batch_loss = 0
                val_acc = 0
                
                # Stored outputs and labels
                outputs_ = []
                labels_ = []
                
                # Validation loop
                D = []
                L = []
                for step, batch in enumerate(tqdm(dataset)) if is_main_process() else enumerate(dataset):
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
                        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
                        
                        # MSE loss if stsb
                        if self.num_to_dataset_name[dataset_name] == "stsb":
                            loss = torch.nn.MSELoss()(outputs, labels.to(outputs.device))
                        # Cross entropy loss otherwise
                        else:
                            loss = torch.nn.CrossEntropyLoss()(outputs, labels.long().to(outputs.device))
                        
                    # Update batch losses
                    val_batch_loss += loss.item()/len(dataset)
                    
                    # Update batch accuracy
                    if self.num_to_dataset_name[dataset_name] == "qqp" or self.num_to_dataset_name[dataset_name] == "mrpc":
                        # F1 score
                        TP = ((outputs.argmax(dim=-1) == labels.long().to(outputs.device)) & (labels == 1)).sum().item()
                        FP = ((outputs.argmax(dim=-1) != labels.long().to(outputs.device)) & (outputs.argmax(dim=-1) == 1)).sum().item()
                        FN = ((outputs.argmax(dim=-1) != labels.long().to(outputs.device)) & (outputs.argmax(dim=-1) == 0)).sum().item()
                        
                        val_acc += TP/(TP + 0.5*(FP+FN))/len(dataset) if TP + 0.5*(FP+FN) != 0 else 0
                    else:
                        outputs_ += list(outputs.squeeze(-1).float().cpu().detach().numpy())
                        labels_ += list(labels.float().cpu().detach().numpy())
                        
                # Calculate the stsb Pearson correlation
                if self.num_to_dataset_name[dataset_name] == "stsb":
                    outputs_ = np.array(outputs_)
                    labels_ = np.array(labels_)
                    # val_acc = 1 - (
                    #         (6*((outputs_ - labels_)**2).sum())
                    #         / (outputs_.shape[0] * (outputs_.shape[0]**2 - 1))
                    #     ) 
                    val_acc = abs(np.corrcoef(outputs_, labels_)[0, 1])
                elif self.num_to_dataset_name[dataset_name] != "qqp" and self.num_to_dataset_name[dataset_name] != "mrpc":
                    val_acc = (np.stack(outputs_).argmax(-1)==np.stack(labels_)).astype(float).mean()
                        
                return val_batch_loss, val_acc
                        
            if self.finetune_task == "mnli":
                val_loss_matched, val_acc_matched = val_loop(val_dataset["matched"])
                val_loss_mismatched, val_acc_mismatched = val_loop(val_dataset["mismatched"])
            else:
                val_batch_loss, val_acc = val_loop(val_dataset)
                
            self.model.train()
                
                
                
                
            # Log wandb
            if is_main_process():
                if self.finetune_task == "mnli":
                    wandb_args = {
                        f"loss_finetune_{self.finetune_task}": batch_loss,
                        f"acc_finetune_{self.finetune_task}": batch_acc,
                        f"val_loss_finetune_{self.finetune_task}_matched": val_loss_matched,
                        f"val_acc_finetune_{self.finetune_task}_matched": val_acc_matched,
                        f"val_loss_finetune_{self.finetune_task}_mismatched": val_loss_mismatched,
                        f"val_acc_finetune_{self.finetune_task}_mismatched": val_acc_mismatched,
                    }
                else:
                    wandb_args = {
                        f"loss_finetune_{self.finetune_task}": batch_loss,
                        f"acc_finetune_{self.finetune_task}": batch_acc,
                        f"val_loss_finetune_{self.finetune_task}": val_batch_loss,
                        f"val_acc_finetune_{self.finetune_task}": val_acc,
                    }
                
                wandb.log(wandb_args)
            
            
            
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
        if self.finetune_:
            self.model = transformers.GPTJForCausalLM.from_pretrained(checkpoint_path.replace(" ", "_"))
        else:
            self.model = transformers.GPTJForCausalLM.from_pretrained(checkpoint_path.replace(" ", "_"))
        
        # Load the config
        config = torch.load(os.path.join(checkpoint_path, "config.pt"))
        if not self.finetune_: # Don't load some config variables if finetuning
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
            self.wandb_id = config["wandb_id"]
        self.attention_type = config["attention_type"]
        
        # Replace all self attention layers (BertSelfAttention) with the cosine attention layer (GPTCosAttention)
        if self.attention_type == "cos":
            for layer in self.model.transformer.h:
                old = layer.attn
                layer.attn = GPTCosAttention(self.model.config).to(layer.attn.q_proj.weight.device)
                
                # Copy weights
                layer.attn.q_proj.weight.data = old.q_proj.weight.data
                if old.q_proj.bias is not None:
                    layer.attn.q_proj.bias.data = old.q_proj.bias.data
                else:
                    layer.attn.q_proj.bias = None
                layer.attn.k_proj.weight.data = old.k_proj.weight.data
                if old.k_proj.bias is not None:
                    layer.attn.k_proj.bias.data = old.k_proj.bias.data
                else:
                    layer.attn.k_proj.bias = None
                layer.attn.v_proj.weight.data = old.v_proj.weight.data
                if old.v_proj.bias is not None:
                    layer.attn.v_proj.bias.data = old.v_proj.bias.data
                else:
                    layer.attn.v_proj.bias = None
                layer.attn.out_proj.weight.data = old.out_proj.weight.data
                if old.out_proj.bias is not None:
                    layer.attn.out_proj.bias.data = old.out_proj.bias.data
                else:
                    layer.attn.out_proj.bias = None
                
                del old
                
            # Load extra params if needed
            self.model.load_state_dict(torch.load(checkpoint_path.replace(" ", "_") + "/pytorch_model.bin", map_location=self.model.transformer.h[0].attn.q_proj.weight.device), strict=False)
            
            # Clear cache
            torch.cuda.empty_cache()
        
        # Load the tokenizer
        self.tokenizer = torch.load(os.path.join(checkpoint_path, "tokenizer.pt"))             
            
            
        # Put the model on the desired device
        if self.dev != "cpu":
            if self.finetune_:
                self.model = self.model.cuda()
                
                self.model_ref = self.model
            else:
                # Initialize the environment
                if not torch.distributed.is_initialized():
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
            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay, eps=1e-7)
            
        # Load checkpoint for optimizer if not finetuning
        else:
            # Load the optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay, eps=1e-7)
            self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), map_location=self.model.device))
            
            # Load the scheduler
            self.scheduler = get_scheduler(self.optimizer, warmup_steps=self.warmup_steps, total_steps=self.num_steps)
            self.scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scheduler.pt"), map_location=self.model.device))
            
        self.pad_token = torch.tensor([self.tokenizer.pad_token_id])
