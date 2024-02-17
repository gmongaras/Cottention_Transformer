import torch
import FastAttention
from FastAttention import forward, backward



class CustomAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V)
        
        if Q.dtype == torch.float32:
            return FastAttention.forward.float32(Q, K, V, 3)
        elif Q.dtype == torch.float16:
            return FastAttention.forward.float16(Q, K, V, 3)
        elif Q.dtype == torch.bfloat16:
            return FastAttention.forward.bfloat16(Q, K, V, 3)
        else:
            raise ValueError("Only float32, float16, and bfloat16 are supported")

    @staticmethod
    def backward(ctx, grad_output):
        # This method receives the gradient of the loss with respect to the output
        # and computes the gradients with respect to the inputs.
        Q, K, V = ctx.saved_tensors

        # # Compute gradients w.r.t inputs. This requires deriving the gradients
        # # based on your specific operation.
        # grad_Q = grad_output * (V.sum(-1, keepdims=True)*K).cumsum(2)
        # grad_K = grad_output * (V.sum(-1, keepdims=True)*Q.flip(2).cumsum(2).flip(2))
        # grad_V = grad_output * (Q.flip(2).cumsum(2).flip(2) * K).sum(-1, keepdims=True).repeat(1, 1, 1, Q.shape[-1])
        
        if Q.dtype == torch.float32:
            temp = FastAttention.backward.float32(Q, K, V, grad_output)
        elif Q.dtype == torch.float16:
            temp = FastAttention.backward.float16(Q, K, V, grad_output)
        elif Q.dtype == torch.bfloat16:
            temp = FastAttention.backward.bfloat16(Q, K, V, grad_output)
        else:
            raise ValueError("Only float32, float16, and bfloat16 are supported")

        return temp, V, K
        
        # # More memory efficient
        # Q_cumsum = Q.flip(2).cumsum(2).flip(2)
        # # Q_cumsum = Q + torch.sum(Q, dim=2, keepdims=True) - torch.cumsum(Q, dim=2)
        # V_sum = V.sum(-1, keepdim=True)
        # # Q_cumsum = Q.cumsum(dim=2)
        # # Q_cumsum = Q - Q_cumsum + Q_cumsum[:, :, -1:]
        # grad_Q = grad_output * (V_sum*K).cumsum(2)
        # grad_K = grad_output * (V_sum*Q_cumsum)
        # grad_V = grad_output * (Q_cumsum*K).sum(dim=-1, keepdim=True)
        
        ## Faster way to compute cumsum
        # cum_sum = torch.cumsum(x, dim=0)             # faster way
        # re_cum_sum = x - cum_sum + cum_sum[-1:None]  # faster way
        # print('[1, 3, 6]', cum_sum)
        # print('[6, 5, 3]', re_cum_sum)


        # Return the gradients in the order of the inputs to forward
        return grad_Q, grad_K, grad_V