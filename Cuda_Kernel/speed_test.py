import torch
from Custom_Kernel import CustomAttention
import timeit

# Normal softmax attention
def Softmax_Attn(Q, K, V, mask):
    QK = torch.matmul(Q, K.transpose(-2, -1))
    QK = QK / torch.sqrt(torch.tensor(256).float())
    QK = QK.masked_fill(mask == 0, float('-inf'))
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, V)
    return output

# Cosine attention - S^2
def Cosine_Attn_S(Q, K, V, mask):
    Q_ = torch.nn.functional.normalize(Q, p=2, dim=-1)
    K_ = torch.nn.functional.normalize(K, p=2, dim=-1)
    QK = torch.matmul(Q_, K_.transpose(-2, -1)).masked_fill(mask == 0, 0)
    output = torch.matmul(QK, V)
    return output

# Cosine attention - d^2
def Cosine_Attn_d(Q, K, V, mask):
    Q_ = torch.nn.functional.normalize(Q, p=2, dim=-1)
    K_ = torch.nn.functional.normalize(K, p=2, dim=-1)

    
    output = CustomAttention.apply(Q_, K_, V)
    
    return output





def time_and_memory(N, H, S, D, funct):
    # Profile the forward and backward methods for each attention mechanism
    Q = torch.randn(N, H, S, D, device='cuda')
    K = torch.randn(N, H, S, D, device='cuda')
    V = torch.randn(N, H, S, D, device='cuda')
    mask = torch.tril(torch.ones(S, S, device='cuda')).unsqueeze(0).unsqueeze(0)
    mask = mask.repeat(N, H, 1, 1)
    
    # Softmax attention
    Q_ = Q.clone().requires_grad_()
    K_ = K.clone().requires_grad_()
    V_ = V.clone().requires_grad_()
    mem_allocated = torch.cuda.memory_allocated()
    mem_cached = torch.cuda.memory_reserved()
    mem_max = torch.cuda.max_memory_allocated()
    print("Forward")
    print(timeit.timeit(lambda: funct(Q_, K_, V_, mask), 
                        number=10))
    del Q_, K_, V_
    print("Backward")
    Q_ = Q.clone().requires_grad_()
    K_ = K.clone().requires_grad_()
    V_ = V.clone().requires_grad_()
    print(timeit.timeit(lambda: torch.autograd.grad(funct(Q_, K_, V_, mask).sum(), 
                                                    (Q_, K_, V_), retain_graph=True), number=10))
    print("Memory")
    print(torch.cuda.memory_allocated() - mem_allocated)
    print(torch.cuda.memory_reserved() - mem_cached)
    print(torch.cuda.max_memory_allocated() - mem_max)
    del Q_, K_, V_, mask, Q, K, V
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()







def main():
    N = 32
    H = 16
    S = 512
    D = 2048//H
    
    # Profile softmax
    print("Softmax")
    time_and_memory(N, H, S, D, Softmax_Attn)
    print()
    
    # Profile cosine S^2
    print("Cosine S^2")
    time_and_memory(N, H, S, D, Cosine_Attn_S)
    print()
    
    # Profile cosine d^2
    print("Cosine d^2")
    time_and_memory(N, H, S, D, Cosine_Attn_d)

    






if __name__ == "__main__":
    main()