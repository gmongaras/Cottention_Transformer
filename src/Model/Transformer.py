import torch
from transformers import AutoTokenizer

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
            "dim": dim,
            "num_layers": num_layers,
            "scale_factor": scale_factor
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