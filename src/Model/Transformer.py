import torch
from transformers import AutoTokenizer
import json

try:
    import sys
    sys.path.append('src/Model')
    
    from Transformer_Block import Transformer_Block
    from PositionalEncodings import PositionalEncoding1D, Summer
except ModuleNotFoundError:
    from src.Model.Transformer_Block import Transformer_Block
    from src.Model.PositionalEncodings import PositionalEncoding1D, Summer


class Transformer(torch.nn.Module):
    def __init__(self, num_layers, dim, scale_factor, distance_type, activation_type):
        super().__init__()
        
        self.num_layers = num_layers
        self.dim = dim
        self.distance_type = distance_type
        self.activation_type = activation_type
        
        # Tokenizer
        self.tokenizer = [AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)]
        
        # Positional encodings
        self.pos_enc = [Summer(PositionalEncoding1D(dim))]
        
        # Embedding
        self.embedding = torch.nn.Embedding(self.tokenizer[0].vocab_size, dim)
        
        # Transformer blocks
        self.encoder_blocks = torch.nn.ModuleList([
            Transformer_Block(dim, scale_factor, decoder=False, distance_type=distance_type, activation_type=activation_type) for _ in range(num_layers)
        ])
        
        # Final layer
        self.final_layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim*scale_factor),
            torch.nn.GELU(),
            torch.nn.Linear(dim*scale_factor, self.tokenizer[0].vocab_size)
        )
        
        
        
        # Defaults for easy model loading
        self.defaults = {
            "num_layers": num_layers,
            "dim": dim,
            "scale_factor": scale_factor,
            "distance_type": distance_type,
            "activation_type": activation_type
        }
        
    def forward(self, X, return_last=False):
        try:
            masks = X["attention_mask"]
        except KeyError:
            masks = None
        if type(masks) is not type(None):
            masks = masks
        X = X["input_ids"]
        
        # Transfer to device
        if type(masks) is not type(None):
            masks = masks.to(self.final_layer[0].weight.device)
        X = X.to(self.final_layer[0].weight.device)
        
        # masks = Y["attention_mask"].to(self.final_layer.weight.device)
        # Y = Y["input_ids"].to(self.final_layer.weight.device)
        
        # Embedding
        X = self.embedding(X) 
        # Y = self.embedding(Y)
        
        # Positional encodings
        X = self.pos_enc[0](X)
        
        # Transformer blocks
        for i in range(self.num_layers):
            X = self.encoder_blocks[i](X, masks=masks)
        # for i in range(self.num_layers):
        #     Y = self.decoder_blocks[i](Y, cond=X, masks=masks)
        
        # Final layer
        if return_last:
            X = self.final_layer(X[:, -1])
        else:
            X = self.final_layer(X)
        
        return X
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Used to load in checkpoints
    def load_checkpoint(self, path):
        # Load in paramaters
        with open(path + "/model_params.json", "r") as f:
            self.defaults = json.load(f)
        
        step = self.defaults["step"]
        epoch = self.defaults["epoch"]
        wandb_id = self.defaults["wandb_id"]
        del self.defaults["step"]
        del self.defaults["epoch"]
        del self.defaults["wandb_id"]
        
        device = self.final_layer[0].weight.device
            
        # Reinit model with checkpoint
        self.__init__(**self.defaults)
        
        # Load in the model
        self.load_state_dict(torch.load(path + "/model.pth", map_location=device))
        self.eval()
        
        # Load in the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        try:
            optimizer.load_state_dict(torch.load(path + "/optimizer.pth", map_location=device))
        except:
            print("Optimizer checkpoint not found")
            optimizer = None
        
        # Load in the scheduler
        if type(optimizer) is not type(None):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-6)
            try:
                scheduler.load_state_dict(torch.load(path + "/scheduler.pth", map_location=device))
            except:
                print("Scheduler checkpoint not found")
                scheduler = None
        else:
            scheduler = None
        
        return optimizer, scheduler, epoch, step, wandb_id