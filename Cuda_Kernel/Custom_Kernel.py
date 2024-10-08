import torch
import FastAttention2 as FastAttention
from FastAttention2 import kernel



class CustomAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        # Ensure Q, K, and V are contiguous
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        
        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V)
        # ctx.save_for_backward(Q.to(torch.float16), K.to(torch.float16), V.to(torch.float16))
        
        inplace=False
        
        # # Mark inputs as dirty
        # ctx.mark_dirty(Q, K, V)
        if Q.dtype == torch.float32:
            return FastAttention.kernel.float32(Q, K, V, 1, inplace, False)
        elif Q.dtype == torch.float64:
            return FastAttention.kernel.float64(Q, K, V, 1, inplace, False)
        elif Q.dtype == torch.float16:
            return FastAttention.kernel.float16(Q, K, V, 1, inplace, False)
        elif Q.dtype == torch.bfloat16:
            return FastAttention.kernel.bfloat16(Q, K, V, 1, inplace, False)
        else:
            raise ValueError("Only float32, float64, float16, and bfloat16 are supported")

    @staticmethod
    def backward(ctx, grad_output):
        # This method receives the gradient of the loss with respect to the output
        # and computes the gradients with respect to the inputs.
        Q, K, V = ctx.saved_tensors
        
        # Ensure the tensors are contiguous
        Q = Q.contiguous()#.to(grad_output.dtype)
        K = K.contiguous()#.to(grad_output.dtype)
        V = V.contiguous()#.to(grad_output.dtype)
        grad_output = grad_output.contiguous()
        
        # if Q.dtype == torch.float32:
        #     temp = FastAttention.backward.float32(Q.clone(), K.clone(), V.clone(), grad_output)
        # elif Q.dtype == torch.float64:
        #     temp = FastAttention.backward.float64(Q.clone(), K.clone(), V.clone(), grad_output)
        # elif Q.dtype == torch.float16:
        #     temp = FastAttention.backward.float16(Q, K, V, grad_output)
        # elif Q.dtype == torch.bfloat16:
        #     temp = FastAttention.backward.bfloat16(Q, K, V, grad_output)
        # else:
        #     raise ValueError("Only float32, float64, float16, and bfloat16 are supported")

        # return temp, V, K
        
        
        """
        # Gradient for Q: ((grad_output @ V) * M) @ K
        # grad_Q = ((grad_output @ V.transpose(-1, -2)) * ~torch.triu(torch.ones(1, 1, Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=Q.device), 1).bool()) @ K
        grad_Q = ((V.unsqueeze(-1)*K.unsqueeze(-2)).cumsum(2) * grad_output.unsqueeze(-1)).sum(-2)
        
        # Gradient for K: (Q.T @ ((grad_output @ V) * M)).T
        # grad_K = (Q.transpose(-1, -2) @ ((grad_output @ V.transpose(-1, -2)) * ~torch.triu(torch.ones(1, 1, Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=Q.device), 1).bool())).transpose(-1, -2)
        # grad_K = ((grad_output @ V.transpose(-1, -2)) * ~torch.triu(torch.ones(1, 1, Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=Q.device), 1).bool()).transpose(-1, -2) @ Q
        grad_K = ((grad_output.unsqueeze(-1)*Q.unsqueeze(-2)).flip(2).cumsum(2).flip(2) * V.unsqueeze(-1)).sum(-2)
        
        # Gradient for V: ((Q@K.T) * M) @ grad_output
        # grad_V = ((Q @ K.transpose(-1, -2)) * ~torch.triu(torch.ones(1, 1, Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=Q.device), 1).bool()).transpose(-1, -2) @ grad_output
        grad_V = (((grad_output.unsqueeze(-1)*Q.unsqueeze(-2)).flip(2).cumsum(2).flip(2) * K.unsqueeze(-2) ).sum(-1))
        """
        
        
        ### More efficient
        
        # # Gradient for Q
        # grad_Q = ((V.unsqueeze(-1)*K.unsqueeze(-2)).cumsum(2) * grad_output.unsqueeze(-1)).sum(-2)
        
        # # Pre-calculate cumsum for grad_output and Q
        # grad_output_cumsum = (grad_output.unsqueeze(-1)*Q.unsqueeze(-2)).flip(2).cumsum(2).flip(2)
        
        # # Gradient for K
        # grad_K = (grad_output_cumsum * V.unsqueeze(-1) ).sum(-2)
        
        # # Gradient for V
        # grad_V = (grad_output_cumsum * K.unsqueeze(-2) ).sum(-1)
        
        
        
        
        
        
        ### Custom implementation
        
        inplace=False
        
        # Gradient for Q - just a forward pass where (Q = grad_output, K = V, V = K)
        if Q.dtype == torch.float32:
            # grad_Q = FastAttention.forward.float32(grad_output, V, K, 1, inplace)
            # grad_K = FastAttention.forward.float32(grad_output, Q, V, 1, inplace)
            # grad_V = FastAttention.forward.float32(grad_output, K, Q, 1, inplace)
            # grad_Q, grad_K, grad_V = FastAttention.backward.float32(Q, K, V, grad_output, 1)
            
            grad_Q = FastAttention.kernel.float32(grad_output, V, K, 1, inplace, False)
            grad_K = FastAttention.kernel.float32(V, grad_output, Q, 1, inplace, True)
            grad_V = FastAttention.kernel.float32(K, Q, grad_output, 1, inplace, True)
        elif Q.dtype == torch.float64:
            # grad_Q = FastAttention.forward.float64(grad_output, V, K, 1, inplace)
            # grad_K = FastAttention.forward.float64(grad_output, Q, V, 1, inplace)
            # grad_V = FastAttention.forward.float64(grad_output, K, Q, 1, inplace)
            # grad_Q, grad_K, grad_V = FastAttention.backward.float64(Q, K, V, grad_output, 1)
            
            grad_Q = FastAttention.kernel.float64(grad_output, V, K, 1, inplace, False)
            grad_K = FastAttention.kernel.float64(V, grad_output, Q, 1, inplace, True)
            grad_V = FastAttention.kernel.float64(K, Q, grad_output, 1, inplace, True)
        elif Q.dtype == torch.float16:
            # grad_Q = FastAttention.forward.float16(grad_output, V, K, 1, inplace)
            # grad_K = FastAttention.forward.float16(grad_output, Q, V, 1, inplace)
            # grad_V = FastAttention.forward.float16(grad_output, K, Q, 1, inplace)
            # grad_Q, grad_K, grad_V = FastAttention.backward.float16(Q, K, V, grad_output, 1)
            
            grad_Q = FastAttention.kernel.float16(grad_output, V, K, 1, inplace, False)
            grad_K = FastAttention.kernel.float16(V, grad_output, Q, 1, inplace, True)
            grad_V = FastAttention.kernel.float16(K, Q, grad_output, 1, inplace, True)
        elif Q.dtype == torch.bfloat16:
            # grad_Q = FastAttention.forward.bfloat16(grad_output, V, K, 1, inplace)
            # grad_K = FastAttention.forward.bfloat16(grad_output, Q, V, 1, inplace)
            # grad_V = FastAttention.forward.bfloat16(grad_output, K, Q, 1, inplace)
            # grad_Q, grad_K, grad_V = FastAttention.backward.bfloat16(Q, K, V, grad_output, 1)
            
            grad_Q = FastAttention.kernel.bfloat16(grad_output, V, K, 1, inplace, False)
            grad_K = FastAttention.kernel.bfloat16(V, grad_output, Q, 1, inplace, True)
            grad_V = FastAttention.kernel.bfloat16(K, Q, grad_output, 1, inplace, True)
        else:
            raise ValueError("Only float32, float64, float16, and bfloat16 are supported")


        # Return the gradients in the order of the inputs to forward
        return grad_Q, grad_K, grad_V
    
    
    
    
# Autograd check
if __name__ == "__main__":
    import torch
    import torch.autograd.profiler as profiler
    import time
    import gc
    import numpy as np
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_printoptions(linewidth=200)
    
    N, H, S, D = 1, 2, 63, 128
    Q = torch.rand(N, H, S, D, requires_grad=True).cuda()
    K = torch.rand(N, H, S, D, requires_grad=True).cuda()
    V = torch.rand(N, H, S, D, requires_grad=True).cuda()
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool), 1).cuda()
    
    # Forward pass
    print("Forward pass")
    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        with profiler.record_function("Method 1"):
            out1 = FastAttention.kernel.float32(Q, K, V, 1, False, False)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print()
    print()
    
    # Backward pass
    print("Backward pass")
    grad_output = torch.rand_like(out1)
    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        with profiler.record_function("Method 1"):
            out1 = FastAttention.kernel.float32(Q, K, V, 1, False, False)
            # grad_Q, grad_K, grad_V = FastAttention.kernel.float32(Q, K, V, grad_output, 1, reversed=True)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print()
    print()
    

    print("Autograd check")
    Q = torch.rand(N, H, S, D, requires_grad=True).cuda()
    K = torch.rand(N, H, S, D, requires_grad=True).cuda()
    V = torch.rand(N, H, S, D, requires_grad=True).cuda()
    check = torch.autograd.gradcheck(CustomAttention.apply, (Q.double(), K.double(), V.double()), eps=1e-4)
    print()