
import torch



# Attention implementation
class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, distance_type="cosine"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.distance_type = distance_type
        
        # query, key, value projections
        self.q_proj = torch.nn.Linear(dim, dim, bias=False)
        self.k_proj = torch.nn.Linear(dim, dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.out_proj = torch.nn.Linear(dim, dim, bias=False)
        
    def forward(self, X, cond=None, masks=None):
        # Create upper traingle masks
        if type(masks) is not type(None):
            masks_orig = masks.clone()
            
            lookahead_masks = torch.triu(torch.ones(X.shape[1], X.shape[1]), diagonal=1).bool()
        
            # Create masks from binary of size (N, S) to float with infinities of size (N, S, S) with a -inf on the column vectors
            pad_masks = (~masks.unsqueeze(-1).repeat(1, 1, X.shape[1]).transpose(-1, -2).bool())
            
            # Combine masks
            masks = (pad_masks.to(X.device) + lookahead_masks.to(X.device)).bool()
        else:
            masks_orig = None
            
            masks = torch.triu(torch.ones(X.shape[1], X.shape[1]), diagonal=1).bool().repeat(X.shape[0], 1, 1).to(X.device)
            
        
        # Q, K, V, projections
        Q = self.q_proj(cond if type(cond) is not type(None) else X)# * masks.unsqueeze(-1)
        K = self.k_proj(cond if type(cond) is not type(None) else X)# * masks.unsqueeze(-1)
        V = self.v_proj(X)# * masks.unsqueeze(-1)
        
        # Split into heads
        Q = Q.reshape(Q.shape[0], Q.shape[1], self.num_heads, -1).transpose(1, 2)
        K = K.reshape(K.shape[0], K.shape[1], self.num_heads, -1).transpose(1, 2)
        V = V.reshape(V.shape[0], V.shape[1], self.num_heads, -1).transpose(1, 2)
        if type(masks) != type(None):
            masks = masks[:, None, :, :]
        
        # Normal softmax attention in one line
        # return torch.softmax((Q @ K.transpose(-1, -2))/torch.sqrt(torch.tensor(Q.shape[-1])), dim=-1) @ V
        
        # Masked softmax attention in one line
        # return (torch.softmax((Q @ K.transpose(-1, -2))/torch.sqrt(torch.tensor(Q.shape[-1])) + (masks*-1e9), dim=-1)) @ V
        
        # Masked attention with no SM
        # return ((Q @ K.transpose(-1, -2))/torch.sqrt(torch.tensor(Q.shape[-1])) + (masks*-1e9)) @ V
        
        # Efficient Cottention?
        # return ((Q)/torch.norm(Q, 2, 2)[:, :, None]) \
        #     @ (((K)/torch.norm(K, 2, 2)[:, :, None]).transpose(-1, -2) \
        #     @ V)
        
        # Inefficient coattention with mask
        # return ((((Q)/torch.norm(Q, 2, 2)[:, :, None]) \
        #     @ ((K)/torch.norm(K, 2, 2)[:, :, None]).transpose(-1, -2))*~masks) \
        #     @ V
            
        # With ReLU
        # return ((((Q)/torch.norm(Q, 2, 2)[:, :, None]) \
        #     @ ((K)/torch.norm(K, 2, 2)[:, :, None]).transpose(-1, -2)).relu()) \
        #     @ V
            
        # Inefficient coattention with mask and relu
        if self.distance_type == "cosine":
            out =  (((((Q)/torch.norm(Q, 2, -1).unsqueeze(-1)) \
                @ ((K)/torch.norm(K, 2, -1).unsqueeze(-1)).transpose(-1, -2)).relu()*(~masks if type(masks) != type(None) else 1)) \
                @ V) * (masks_orig[:, None, :, None] if type(masks_orig) != type(None) else 1)
        # normalized L2 attention
        elif self.distance_type == "l2":
            raise NotImplementedError("L2 distance not implemented")
            out =  ((Q@K.T)*(~masks if type(masks) != type(None) else 1) \
                @ V) * (masks_orig[:, None, :, None] if type(masks_orig) != type(None) else 1)
        else:
            raise ValueError("distance_type must be either 'cosine' or 'l2'")
        
        
        
        # Merge heads
        out = out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
        
        # Output projection
        return self.out_proj(out)
        
        
        
        
        
        
        
if __name__ == "__main__":
    # Dummy tensor of shape 1xSxd
    N = 1
    S = 20
    d = 5
    X = torch.randn(1, S, d)
    
    # Attention
    attn = Attention(d)
    
    # Mask half of the sequence 
    masks = torch.zeros(1, S).bool()
    masks[:, :10] = True
    
    # Get gradient of attention outptus with respect to input
    X.requires_grad = True
    Y = attn(X, masks=masks)*masks[:, :, None]
    # Get X gradient
    grad = torch.autograd.grad(Y.sum(), X)[0][:, :10]
    
    
    # Gradient of attention outputs with respect to input without mask
    X = X[:, :10, :].detach()
    X.requires_grad = True
    Y2 = attn(X)
    # Get X gradient
    grad2 = torch.autograd.grad(Y2.sum(), X)[0]
    
    # Should be equal
    print(torch.allclose(grad, grad2))