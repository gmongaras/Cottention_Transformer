import torch

try:
    import sys
    sys.path.append('src/Model')
    
    from Attention import Attention
except ModuleNotFoundError:
    from src.Model.Attention import Attention
            
            
            
            
# Transformer
class Transformer_Block(torch.nn.Module):
    def __init__(self, dim, scale_factor=4, decoder=False, distance_type="cosine", activation_type="relu"):
        super().__init__()
        
        self.dim = dim
        self.decoder = decoder
        
        self.LN1 = torch.nn.LayerNorm(dim)
        self.LN2 = torch.nn.LayerNorm(dim)
        self.attn = Attention(dim, distance_type=distance_type)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(dim, int(dim*scale_factor)),
            torch.nn.GELU(),
            torch.nn.Linear(int(dim*scale_factor), dim)
        )
        
        if decoder:
            self.LN3 = torch.nn.LayerNorm(dim)
            self.attn2 = Attention(dim, distance_type=distance_type, activation_type=activation_type)
        
    def forward(self, X, cond=None, masks=None):
        # Residual
        res = X.clone()
        
        # Layer norm
        X = self.LN1(X)
        
        # Attention
        X = self.attn(X, masks=masks) + res
        
        # Cross attention
        if self.decoder:
            # Residual
            res = X.clone()
            
            # Layer norm
            X = self.LN3(X)
            
            # Attention
            X = self.attn2(X, cond=cond) + res
        
        # Residual
        res = X.clone()
        
        # Layer norm 2
        X = self.LN2(X)
        
        # Feed forward
        X = self.ff(X) + res
        
        return X