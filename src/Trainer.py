import torch
import os
from tqdm import tqdm
import json
import pytorch_warmup as warmup
from contextlib import nullcontext
import wandb





from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

try:
    import sys
    sys.path.append('src/utils')
    
    from multi_gpu_helpers import is_main_process
except ModuleNotFoundError:
    from src.utils.multi_gpu_helpers import is_main_process








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
    
    
    
# def collate_fn(batch):
#     return batch
    
    
class Trainer():
    def __init__(self, 
            model, 
            dataset,
            dev="cpu", 
            batch_size=32, 
            num_workers=0, 
            prefetch_factor=1, 
            lr=1e-3, 
            save_every_steps=1000, 
            use_scheduler=True, 
            checkpoints_dir="checkpoints", 
            accumulation_steps=1, 
            optimizer_checkpoint=None, 
            scheduler_checkpoint=None, 
            use_amp=True,
            clipping_value=None,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.999,
            warmup_steps=1000,
            wandb_name=None,
        ):
        self.model = model
        self.dev = dev
        self.lr = lr
        self.save_every_steps = save_every_steps
        self.use_scheduler = use_scheduler
        self.checkpoints_dir = checkpoints_dir
        self.accumulation_steps = accumulation_steps
        self.optimizer_checkpoint = optimizer_checkpoint
        self.scheduler_checkpoint = scheduler_checkpoint
        self.use_amp = use_amp
        self.clipping_value = clipping_value
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.warmup_steps = warmup_steps
        self.wandb_name = wandb_name
        
        
        
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
        
        
        
        # Distributed dataloader
        if self.dev == "cpu":
            self.dataloader = DataLoader(dataset, 
                batch_size=batch_size,
                shuffle=True,  
                # collate_fn=lambda x: x,
                # collate_fn=collate_fn,
                num_workers=num_workers if num_workers > 0 else 0,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
            )
        else:
            self.dataloader = DataLoader(dataset, 
                batch_size=batch_size, 
                # collate_fn=lambda x: x,
                # collate_fn=collate_fn,
                num_workers=num_workers if num_workers > 0 else 0,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
                sampler=DistributedSampler(dataset, shuffle=True)
            )
            
            
            
    def train(self, epoch_ckpt=None, step_ckpt=None):
        self.model.train()
        
        # Initialize wandb run
        if is_main_process():
            wandb.init(
                project="Cottention",
                name=self.wandb_name,
            )
            wandb.watch(self.model, log_freq=self.save_every_steps)
        
        if epoch_ckpt is None:
            epoch_ckpt = 0
        if step_ckpt is None:
            step_ckpt = 1
            
        # Model reference is different depending on the device
        if self.dev == "cpu":
            model_ref = self.model
            
            # World size is 1 if not distributed
            world_size = 1
        else:
            model_ref = self.model.module
            
            # Get world size
            world_size = int(os.environ['WORLD_SIZE'])
        
        # Optimzer
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                    lr=self.lr, 
                    weight_decay=self.weight_decay, 
                    betas=(self.adam_beta1, self.adam_beta2),
                    eps=1e-7 if self.use_amp else 1e-8)\
                        if self.optimizer_checkpoint is None else self.optimizer_checkpoint
        
        # Loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        # Cosine annealing scheduler with warm restarts
        scheduler = None
        if self.use_scheduler:
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10000*len(self.dataloader), eta_min=1e-6) if self.scheduler_checkpoint is None else self.scheduler_checkpoint
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=1e-6) if self.scheduler_checkpoint is None else self.scheduler_checkpoint
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10000*len(self.dataloader), eta_min=1e-6) if self.scheduler_checkpoint is None else self.scheduler_checkpoint
            # https://github.com/Tony-Y/pytorch_warmup
            # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
            warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=self.warmup_steps)
            
        if self.use_amp:
            grad_scaler = torch.cuda.amp.GradScaler()
        
        num_steps = step_ckpt
        for epoch in range(epoch_ckpt, 10000):
            # Set the epoch number for the dataloader to seed the
            # randomization of the sampler
            if self.dev != "cpu":
                self.dataloader.sampler.set_epoch(epoch)
            
            # Total batch loss
            batch_loss = 0
            
            # Iterate over the batches of data
            for batch_num, batch in enumerate(tqdm(self.dataloader)):
                with torch.no_grad():
                    # Model reference is different depending on the device
                    if self.dev == "cpu":
                        model_ref = self.model
                    else:
                        model_ref = self.model.module
                        
                # Convert batch to torch tensors
                batch["input_ids"] = torch.stack(batch["input_ids"]).T
                batch["labels"] = torch.stack(batch["labels"]).T
                batch["attention_mask"] = torch.stack(batch["attention_mask"]).T
                    
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                    # Send through model
                    output = self.model_ref(batch)
                    
                    # Calculate loss
                    loss = loss_fn(output.transpose(-1, -2), batch["labels"].to(output.device))
                    
                    # Divide loss by accumulation steps
                    loss = loss / self.accumulation_steps
                    
                    # Backward pass
                    if self.use_amp:
                        grad_scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    # Update model every accumulation_steps 
                    if num_steps % self.accumulation_steps == 0:
                        # Clip gradients
                        if self.use_amp:
                            grad_scaler.unscale_(optimizer)
                        if self.clipping_value is not None:
                            torch.nn.utils.clip_grad_norm_(model_ref.parameters(), self.clipping_value)
                        
                        
                        # Step scheduler
                        if self.use_scheduler:
                            with warmup_scheduler.dampening():
                                scheduler.step() # Normal
                                # scheduler.step(loss) # Plateau
                                # scheduler.step(epoch + batch_num / len(self.dataloader)) # Cosine Annealing
                        # Step optimizer
                        if self.use_amp:
                            grad_scaler.step(optimizer)
                        else:
                            optimizer.step()
                        # Update scaler for next iteration
                        if self.use_amp:
                            grad_scaler.update()
                        optimizer.zero_grad()
                    
                    if num_steps-step_ckpt < self.save_every_steps and is_main_process():
                        print(f"Step: {num_steps} | Loss: {loss.item()} | Perplexity: {loss.exp().item()}")
                    
                    batch_loss += loss.item()
                    num_steps += 1
                
                
                
                
                
                
                
                # Save audio samples
                if num_steps % self.save_every_steps == 0 and is_main_process():
                    with torch.no_grad():
                        if is_main_process():
                            wandb.log({"loss": batch_loss/self.save_every_steps})
                            wandb.log({"perplexity": torch.tensor(batch_loss/self.save_every_steps).exp()})
                            wandb.log({"lr": optimizer.param_groups[0]['lr']})
                            wandb.log({"step": num_steps})
                            wandb.log({"epoch": epoch})
                        
                        print(f"Step: {num_steps} | Loss: {batch_loss/self.save_every_steps} | Perplexity: {torch.tensor(batch_loss/self.save_every_steps).exp()}")
                        batch_loss *= 0

                        # Save model parameters to json
                        if not os.path.exists(f"{self.checkpoints_dir}/step_{num_steps}"):
                            os.makedirs(f"{self.checkpoints_dir}/step_{num_steps}")
                        with open(f"{self.checkpoints_dir}/step_{num_steps}/model_params.json", "w") as f:
                            model_ref.defaults["step"] = num_steps+1
                            model_ref.defaults["epoch"] = epoch
                            json.dump(model_ref.defaults, f)
                        
                        # Save model checkpoints
                        if not os.path.exists(f"{self.checkpoints_dir}/step_{num_steps}"):
                            os.makedirs(f"{self.checkpoints_dir}/step_{num_steps}")
                        torch.save(model_ref.state_dict(), f"{self.checkpoints_dir}/step_{num_steps}/model.pth")
                        
                        # Save optimizer checkpoints
                        if not os.path.exists(f"{self.checkpoints_dir}/step_{num_steps}"):
                            os.makedirs(f"{self.checkpoints_dir}/step_{num_steps}")
                        torch.save(optimizer.state_dict(), f"{self.checkpoints_dir}/step_{num_steps}/optimizer.pth")
                        
                        # Save scheduler checkpoints
                        if self.use_scheduler:
                            if not os.path.exists(f"{self.checkpoints_dir}/step_{num_steps}"):
                                os.makedirs(f"{self.checkpoints_dir}/step_{num_steps}")
                            torch.save(scheduler.state_dict(), f"{self.checkpoints_dir}/step_{num_steps}/scheduler.pth")
                
                
                
                
                
                
                
                
                # Free memory
                del batch, output, loss
                
            if is_main_process():
                print(f"Epoch: {epoch} | Batch Loss: {batch_loss/len(self.dataloader)} | Perplexity: {torch.tensor(batch_loss/len(self.dataloader)).exp()}")