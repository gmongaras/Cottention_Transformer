import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableLPDistance(nn.Module):
    def __init__(self, initial_p=2.0, epsilon=1e-10):
        super(LearnableLPDistance, self).__init__()
        
        # Initialize p as a learnable parameter
        self.p_raw = nn.Parameter(torch.tensor([initial_p], dtype=torch.float32, requires_grad=True))
        self.epsilon = epsilon

    def forward(self, A, B):
        # Apply Softplus to ensure p is positive
        p = F.softplus(self.p_raw) + self.epsilon
        
        # Compute L-p norms along the last dimension (E)
        A_norm = torch.pow(torch.sum(torch.pow(torch.abs(A), p), dim=-1, keepdim=True), 1/p) + self.epsilon
        B_norm = torch.pow(torch.sum(torch.pow(torch.abs(B), p), dim=-1, keepdim=True), 1/p) + self.epsilon
        # A_norm = torch.norm(A, p=p.item(), dim=-1, keepdim=True) + self.epsilon
        # B_norm = torch.norm(B, p=p.item(), dim=-1, keepdim=True) + self.epsilon
        
        # Normalize A and B
        A_normalized = A / A_norm
        B_normalized = B / B_norm
        
        # Expand dimensions for broadcasting
        A_expanded = A_normalized.unsqueeze(3)
        B_expanded = B_normalized.unsqueeze(2)
        
        # Compute L-p distance between all combinations of vectors in A and B
        distance = torch.pow(torch.sum(torch.pow(torch.abs(A_expanded - B_expanded), p), dim=-1), 1/p)
        # distance = torch.norm(A_expanded - B_expanded, p=p.item(), dim=-1)
        
        # Output is in the range [0, 2]
        # put it in the range [0, 1] and invert it
        # so higher scores measure higher similarity
        distance = 1 - distance / 2
        
        return distance
