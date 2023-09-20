import torch
from Transformer_Block import Transformer_Block
from transformers import AutoTokenizer




class Transformer(torch.nn.Module):
    def __init__(self, num_layers, dim):
        super().__init__()
        
        self.num_layers = num_layers
        self.dim = dim
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        
        # Embedding
        self.embedding = torch.nn.Embedding(self.tokenizer.vocab_size, dim)
        
        # Transformer blocks
        self.encoder_blocks = torch.nn.ModuleList([
            Transformer_Block(dim) for _ in range(num_layers)
        ])
        # self.decoder_blocks = torch.nn.ModuleList([
        #     Transformer_Block(dim, decoder=True) for _ in range(num_layers)
        # ])
        
        # Final layer
        self.final_layer = torch.nn.Linear(dim, self.tokenizer.vocab_size)
        
    def forward(self, X):
        masks = X["attention_mask"].to(self.final_layer.weight.device)
        X = X["input_ids"].to(self.final_layer.weight.device)
        # masks = Y["attention_mask"].to(self.final_layer.weight.device)
        # Y = Y["input_ids"].to(self.final_layer.weight.device)
        
        # Embedding
        X = self.embedding(X) 
        # Y = self.embedding(Y)
        
        # Transformer blocks
        for i in range(self.num_layers):
            X = self.encoder_blocks[i](X, masks=masks)
        # for i in range(self.num_layers):
        #     Y = self.decoder_blocks[i](Y, cond=X, masks=masks)
        
        # Final layer
        X = self.final_layer(X)
        
        return X