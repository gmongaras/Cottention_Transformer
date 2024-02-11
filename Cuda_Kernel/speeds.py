import torch
import custom_op
from Custom_Kernel import CustomAttention

N = 256
S = 256
D = 32

# Method 1
def method1(Q, K, V, mask):
    QK = torch.matmul(Q, K.transpose(-2, -1))
    QK = QK / torch.sqrt(torch.tensor(D).float())
    QK = QK.masked_fill(mask == 0, float('-inf'))
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, V)
    return output

# Method 2
def method2(Q, K, V, mask):
    Q_ = Q#torch.nn.functional.normalize(Q, p=2, dim=-1)
    K_ = K#torch.nn.functional.normalize(K, p=2, dim=-1)
    QK = torch.matmul(Q_, K_.transpose(-2, -1)).masked_fill(mask == 0, 0)
    output = torch.matmul(QK, V)
    return output
def method2_(Q, K, V, mask):
    QK = torch.matmul(Q, K.transpose(-2, -1)).masked_fill(mask == 0, 0)
    output = torch.matmul(QK, V)
    return output

# # Method 3
# def method3(Q, K, V, mask):
#     VK = [] # calculate the mat muls first
#     V_ = V.unsqueeze(-1).transpose(1, 2)
#     K_ = K.unsqueeze(1)
#     for i in range(0, S):
#         VK.append(
#             # torch.einsum("bdS1,b1Sd->bd", Q, VK)
#             (V_[:, :, :i+1, :]*K_[:, :, :i+1, :]).sum(2)
#             )
#     VK = torch.stack(VK)
#     return torch.einsum("bsD,sbdD->bsd", Q, VK)

# Method 3
def method3(Q, K, V, mask):
    VK = [] # calculate the mat muls first
    V_ = V.transpose(1, 2)
    for i in range(0, S):
        VK.append(
            torch.einsum("bDS,bSd->bDd", V_[:, :, :i+1], K[:, :i+1, :])
            )
    VK = torch.stack(VK)
    output = torch.einsum("bsD,sbdD->bsd", Q, VK)
    return output


# Method 4
def method4(Q, K, V, mask):
    VK = torch.zeros(S, N, D, D).cuda()
    V_ = V.unsqueeze(-1).transpose(1, 2)
    K_ = K.unsqueeze(1)
    VK_ = V_ * K_
    for i in range(0, S):
        VK[i] = VK_[:, :, :i+1, :].sum(2)
    output = torch.einsum("bsD,sbdD->bsd", Q, VK)
    return output

# Method 5
def method5(Q, K, V, mask):
    # VK = (V.unsqueeze(-1).transpose(1, 2) * K.unsqueeze(1)).cumsum(2)
    # return torch.einsum("bsD,bdsD->bsd", Q, VK)
    
    VK = (V.unsqueeze(-1) * K.unsqueeze(2)).cumsum(1)
    output = torch.einsum("bsD,bsdD->bsd", Q, VK)
    return output
    # return (VK * Q.unsqueeze(-2)).sum(-1)


# # Method 6
# import torch.nn.functional as F
# def cumulative_sum_via_convolution(V, K):
#     # Element-wise multiplication
#     VK = V.unsqueeze(-1).transpose(1, 2) * K.unsqueeze(1)
    
#     # Getting the shape for creating the ones tensor
#     sequence_length = VK.size(2)
    
#     # Convolution requires input of shape (batch, channel, length), which VK already is
#     # The kernel should be of shape (out_channels, in_channels, kernel_size)
#     # Since we're simulating cumsum, in_channels = out_channels = 1 and kernel_size = sequence_length
#     # Padding is set to 'same' to keep output dimensions consistent with input
#     ones_kernel = torch.ones((1, 1, sequence_length), device=VK.device, dtype=VK.dtype)
    
#     # Apply 1D convolution with padding to keep the original length
#     # Stride is 1 and padding is set to maintain the output size
#     # The cumulative sum is achieved by using a very wide convolutional window that covers all previous elements
#     cumsum = F.conv1d(VK, ones_kernel, padding='same', groups=VK.size(1))
    
#     return cumsum
# def method6(Q, K, V, mask):
#     VK = cumulative_sum_via_convolution(V, K)
#     return torch.einsum("bsD,bdsD->bsd", Q, VK)

# Method 6 - custom op
Attn = CustomAttention.apply
def method6(Q, K, V, mask):
    Q = Q.unsqueeze(1)
    K = K.unsqueeze(1)
    V = V.unsqueeze(1)
    # output = torch.empty_like(Q).cuda()  # Prepare an output tensor
    # custom_op.compute_and_contract(Q, K, V, output)
    # print()
    
    output = CustomAttention.apply(Q, K, V).squeeze(1)
    return output
    
    # VK = (V.unsqueeze(-1).transpose(1, 2) * K.unsqueeze(1)).cumsum(2)
    # return torch.einsum("bsD,bdsD->bsd", Q, VK)
    
    # VK = (K.unsqueeze(-2) * V.unsqueeze(-1)).cumsum(1)
    # VK = (V.cumsum(1).unsqueeze(-1) * K.cumsum(1).unsqueeze(2))
    # return torch.einsum("bsD,bsdD->bsd", Q, VK)
    
    # QV = (Q.unsqueeze(-1)*V.unsqueeze(-2))
    # QVK = QV*(K.cumsum(1).unsqueeze(-1))
    # out = QVK.sum(-2)
    
    # print()
    # return (VK * Q.unsqueeze(-2)).sum(-1)


# Time the methods
import timeit
Q, K, V = torch.rand(N, S, D).cuda(), torch.rand(N, S, D).cuda(), torch.rand(N, S, D).cuda()
mask = torch.tril(torch.ones(S, S)).view(1, S, S).cuda()
# print(timeit.timeit(lambda: method1(Q, K, V, mask), number=1000))
# print(timeit.timeit(lambda: method2(Q, K, V, mask), number=1000))
# # print(timeit.timeit(lambda: method3(Q, K, V, mask), number=1000))
# print(timeit.timeit(lambda: method5(Q, K, V, mask), number=1000))

# Time each method using time
method1(Q, K, V, mask)
import time
start = time.time()
method1(Q, K, V, mask)
print("Method 1:", time.time() - start)

start = time.time()
out = method2(Q, K, V, mask)
print("Method 2:", time.time() - start)

start = time.time()
out2 = method5(Q, K, V, mask)
print("Method 5:", time.time() - start)

start = time.time()
method6(Q, K, V, mask)
print("Method 6:", time.time() - start)

# Using torch get the GPU memory usage of each function
import torch
import gc
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())
print(torch.cuda.max_memory_allocated())
print()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

method1(Q, K, V, mask)
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())
print(torch.cuda.max_memory_allocated())
print()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

method2(Q, K, V, mask)
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())
print(torch.cuda.max_memory_allocated())
print()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

# method5(Q, K, V, mask)
# print(torch.cuda.memory_allocated())
# print(torch.cuda.memory_cached())
# print(torch.cuda.max_memory_allocated())
# print()
# torch.cuda.empty_cache()
# gc.collect()
# torch.cuda.reset_peak_memory_stats()

method6(Q, K, V, mask)
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())
print(torch.cuda.max_memory_allocated())






# Testing backward pass
print("\n\n\n\n")
# Method 2
Q, K, V = torch.rand(N, S, D, requires_grad=True).cuda(), torch.rand(N, S, D, requires_grad=True).cuda(), torch.rand(N, S, D, requires_grad=True).cuda()
Q2 = Q.clone().detach().requires_grad_(True)
K2 = K.clone().detach().requires_grad_(True)
V2 = V.clone().detach().requires_grad_(True)
method2(Q2, K2, V2, mask).sum().backward()
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())
print(torch.cuda.max_memory_allocated())
print()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

# Method 6
Q6 = Q.clone().detach().requires_grad_(True)
K6 = K.clone().detach().requires_grad_(True)
V6 = V.clone().detach().requires_grad_(True)
method6(Q6, K6, V6, mask).sum().backward()
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())
print(torch.cuda.max_memory_allocated())
print()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.reset_peak_memory_stats()

assert torch.allclose(Q2.grad, Q6.grad)
assert torch.allclose(K2.grad, K6.grad)
assert torch.allclose(V2.grad, V6.grad)


# Testing speed of backward pass
print("\n\n\n\n")
# Method 2
Q, K, V = torch.rand(N, S, D, requires_grad=True).cuda(), torch.rand(N, S, D, requires_grad=True).cuda(), torch.rand(N, S, D, requires_grad=True).cuda()
Q2 = Q.clone().detach().requires_grad_(True)
K2 = K.clone().detach().requires_grad_(True)
V2 = V.clone().detach().requires_grad_(True)
start = time.time()
method2(Q2, K2, V2, mask).sum().backward()
print("Method 2:", time.time() - start)

# Method 6
Q6 = Q.clone().detach().requires_grad_(True)
K6 = K.clone().detach().requires_grad_(True)
V6 = V.clone().detach().requires_grad_(True)
start = time.time()
method6(Q6, K6, V6, mask).sum().backward()
print("Method 6:", time.time() - start)

print()