import torch
import FastAttention



class CustomAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        output = FastAttention.compute_and_contract(Q, K, V, 5)

        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, output)
        
        return output

    @staticmethod
    @torch.no_grad()
    def backward(ctx, grad_output):
        # This method receives the gradient of the loss with respect to the output
        # and computes the gradients with respect to the inputs.
        Q, K, V, output = ctx.saved_tensors

        # # Compute gradients w.r.t inputs. This requires deriving the gradients
        # # based on your specific operation.
        # grad_Q = grad_output * (V.sum(-1, keepdims=True)*K).cumsum(2)
        # grad_K = grad_output * (V.sum(-1, keepdims=True)*Q.flip(2).cumsum(2).flip(2))
        # grad_V = grad_output * (Q.flip(2).cumsum(2).flip(2) * K).sum(-1, keepdims=True).repeat(1, 1, 1, Q.shape[-1])
        
        # More memory efficient
        Q_cumsum = Q.flip(2).cumsum(2).flip(2)
        # Q_cumsum = Q + torch.sum(Q, dim=2, keepdims=True) - torch.cumsum(Q, dim=2)
        V_sum = V.sum(-1, keepdim=True)
        # Q_cumsum = Q.cumsum(dim=2)
        # Q_cumsum = Q - Q_cumsum + Q_cumsum[:, :, -1:]
        grad_Q = grad_output * (V_sum*K).cumsum(2)
        grad_K = grad_output * (V_sum*Q_cumsum)
        grad_V = grad_output * (Q_cumsum * K).sum(dim=-1, keepdim=True)
        
        ## Faster way to compute cumsum
        # cum_sum = torch.cumsum(x, dim=0)             # faster way
        # re_cum_sum = x - cum_sum + cum_sum[-1:None]  # faster way
        # print('[1, 3, 6]', cum_sum)
        # print('[6, 5, 3]', re_cum_sum)


        # Return the gradients in the order of the inputs to forward
        return grad_Q, grad_K, grad_V