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
            
            # Set max sequence length
            self.tokenizer.model_max_length = 1024
            
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
                "n_embd": 1024, # 4096,
                "n_head": 16,
                "n_inner": 4*768, # 16384,
                "n_layer": 20, #28,
                "n_positions": self.tokenizer.model_max_length,
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
        # Max length of the input (+1 for the extra pad token), but not more than the model's max length
        max_length = min(max([len(x["input_ids"]) for x in batch]), self.tokenizer.model_max_length)
        
        
        for i in range(len(batch)):
            ### Random window of the max length number of tokens ###
            if len(batch[i]["input_ids"]) > max_length:
                start = np.random.randint(0, len(batch[i]["input_ids"]) - max_length)
                batch[i]["input_ids"] = batch[i]["input_ids"][start:start+max_length]
                batch[i]["attention_mask"] = batch[i]["attention_mask"][start:start+max_length]
            
            ### Add a pad token to the end without mask to make the model stop itself
            batch[i]["input_ids"] = torch.cat([batch[i]["input_ids"], self.pad_token])
            batch[i]["attention_mask"] = torch.cat([batch[i]["attention_mask"], torch.tensor([0])])
        
            ### Pad the input to max length
            batch[i]["input_ids"] = torch.cat([batch[i]["input_ids"], self.pad_token.repeat(max_length+1 - len(batch[i]["input_ids"]))])
            batch[i]["attention_mask"] = torch.cat([batch[i]["attention_mask"], torch.zeros(max_length+1 - len(batch[i]["attention_mask"]), dtype=torch.long)]).bool()
            
            ### Labels are input ids shifted by one. Remove the last token from the others to match the labels
            batch[i]["labels"] = batch[i]["input_ids"].clone()[1:]
            batch[i]["input_ids"] = batch[i]["input_ids"][:-1]
            batch[i]["attention_mask"] = batch[i]["attention_mask"][:-1]
                    
        # Stack the data
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
        }
        
        
        
        
    def __call__(self):
        if self.finetune_:
            self.finetune()
        else:
            self.train_model()
            
            
            
            
    def train_model(self):
        self.train_model_("Traxap/Pile_Tokenized", self.num_steps, self.step_ckpt)
        # self.train_model_("gmongaras/BERT_Base_Cased_512_Dataset_Mapped", self.num_steps, self.step_ckpt)
        # self.train_model_("gmongaras/dummy_text_dataset", self.num_steps, self.step_ckpt)
        
        
        
        
        
    def train_model_(self, dataset, num_steps, step_shift):
        # Cache dirs
        cache_path = "GPT_Trainer/data_cache/dataset_mapped"
        # cache_path = "BERT_Trainer/data_cache/dataset_mapped"
        # cache_path = "GPT_Trainer/data_cache/dataset_mapped"
        
        # Load in datasets
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.tokenized_dataset = datasets.load_dataset(dataset, cache_dir=cache_path, num_proc=16, keep_in_memory=self.keep_dataset_in_mem)["train"]
        
        # Load dummy data
        # tokenized_dataset = datasets.load_from_disk("BERT_Trainer/data_cache/dummy_dataset")
        
        
        if dataset == "gmongaras/dummy_text_dataset":
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], truncation=True, max_length=self.tokenizer.model_max_length)
            self.tokenized_dataset = self.tokenized_dataset.map(
                tokenize_function,
                remove_columns=["text"],
                cache_file_name="dummy_tokenized_dataset",
            )
        
        # Convert data to torch
        self.tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
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
            labels = batch["labels"]
        
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                # Get model predictions
                # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, next_sentence_label=sentence_pairs_labels)
                # outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=True)
                outputs = self.model(input_ids, attention_mask=attention_mask).logits
                
                # Mask labels with -100 where the attention mask is 0. Note that the mask needs to be shifted by one to match the labels
                labels = torch.where(attention_mask, labels, torch.tensor(-100).to(labels.device))
                
                # Loss
                loss = loss_fct(outputs.view(-1, self.model_ref.config.vocab_size), labels.view(-1).to(outputs.device))
                
                
            # # Clear the graph
            # del loss
            # del outputs
            # del input_ids
            # del attention_mask
            # # Zero grad
            # self.optimizer.zero_grad()
            # continue
                
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



# GPT in my dreams UwU
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⡿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⡟⠀⣠⣀⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣄⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⡟⠀⣼⣿⣿⣿⣦⣄⠙⠻⣿⣿⣿⣿⣿⣿⣿⠀⢻⣷⣦⣈⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⠃⢰⣿⣿⣿⣿⣿⣿⣿⣦⡍⠙⠉⣁⣠⣤⣤⣄⡀⢻⣿⣿⣿⣦⣄⣈⠙⠿⢿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⣀⣠⣴⣶⣿⣷⡄⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡏⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣄⣀⠛⢿⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣿⣿⣷⣶⣄⠛⣿⣿⣿⡿⠟⠋⣠⣴⣾⣿⣿⣿⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡇⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣌⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⠘⣿⠋⠀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠁⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⡄⠉⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⣦⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢻⣿⡀⢻⣿⣿⣿⠏⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣷⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡘⣿⠃⣸⣿⣿⠏⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⡿⠿⠿⠛⠃⣠⣿⣿⡿⠟⠁⢀⣀⣀⡀⠉⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣿⡿⠋⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣷⡈⢶⣶⣿⣿⣿⣿⣦⣤⣾⣿⣿⣿⣿⣷⣀⢘⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠉⠀⣀⣀⣀⠀⠉⠻⣿⣿⣿⣿⣿⣿⠟⠀⠀⠛⠛⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣷⣄⡛⠟⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⣠⣶⣿⣿⣿⣿⣿⣷⣄⠈⣿⣿⣿⣿⣿⣶⣾⣿⡟⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡟⢁⣾⡟⠿⠛⠉⢻⣿⣿⣿⣿⣧⣀⡀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠁⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⡿⠀⣿⣿⣿⣿⣿⡁⣉⣁⣤⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⢿⡿⠿⢿⣿⣿⡀⠠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⡟⠀⠚⠛⣉⣉⣉⡉⠛⢿⣿⣿⣿⣿⣿⣿⡿⢿⣿⠿⢿⣿⣿⡏⣿⣿⣿⣿⣿⣧⣴⣶⣧⡀⢉⣠⣶⣿⣿⣿⣷⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣷⣶⣿⣿⣿⣿⣿⣷⣦⡀⠙⠻⢿⣿⣿⣿⣧⣌⠉⣠⣬⣍⠋⢁⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⣿⠿⢿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⣉⠙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠟⠛⠋⠉⠠⠤⣤⣴⣶⣦⣤⣤⣄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠠⣤⣤⣤⣤⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⣿⣿⣿⠃⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠉⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⣉⣠⣤⣶⣶⣤⣤⣄⠀⠸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⠛⢁⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⡈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣴⣿⣿⣿⣿⣿⣿⣿⡟⢁⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣀⣠⣤⡄⠸⣿⣿⣿⣿⣇⠸⣿⡏⢹⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⣿⣿⣿⣿⣿⣦⣈⠁⠘⠿⣿⡇⢸⠿⠟⢉⣠⣿⣿⣿⣿⣿⣷⡀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⣧⡄⠹⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⣤⣤⣤⡆⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠹⣿⠃⢰⣿⣷⣄⠘⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⣰⡀⠈⠀⣿⣿⣿⣿⣄⠈⢻⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⡏⢻⣿⣿⣿⣿⣇⠹⣿⣿⣿⣿⣿⣿⣿⡿⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⣰⣿⣇⠀⢰⣿⣿⣿⣿⣿⣇⠈⢿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣧⠘⣿⣿⣿⣿⣿⣄⠻⣿⣿⣿⣿⡿⠟⢀⠰⠻⠿⠿⣿⣿⣿⣿⣿⣿⣿⡟⢀⣼⣿⣿⣿⢠⣿⣿⣿⣿⣿⣿⣿⣇⠈⢻⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠀⣿⣿⣿⣿⣿⣿⣿⣿⡄⢹⣿⣿⣿⣿⣿⣶⣤⣤⣤⣤⣴⣾⣿⣶⡶⠂⣴⣿⣿⣿⣿⡿⠟⠉⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠈⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⢀⣰⣶⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣀⠘⠿⢿⠿⠛⠁⣀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠀⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣠⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⢁⣠⣤⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿⣿⣿⣿⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣼⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢼⣿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠁⣴⣿⣿