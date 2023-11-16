import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# @torch.no_grad()
def time_this(N=100, S=512, E=768):
    Q = torch.randn(N, S, E).cuda()
    K = torch.randn(N, S, E).cuda()
    V = torch.randn(N, S, E).cuda()
    
    # Arrays to store results
    times = []
    peak_memory = []
    
    # Initialize CUDA events
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # First operation (S2)
    start.record()
    torch.cuda.reset_peak_memory_stats()
    result_direct = (Q @ K.transpose(-1, -2)) @ V
    end.record()
    torch.cuda.synchronize()
    # print(f"mathmul S2: {start.elapsed_time(end)} ms, Peak memory: {torch.cuda.max_memory_allocated()} bytes")
    peak_memory.append(torch.cuda.max_memory_allocated())
    times.append(start.elapsed_time(end))
    del result_direct
    torch.cuda.empty_cache()

    # Second operation (d2)
    start.record()
    torch.cuda.reset_peak_memory_stats()
    result_direct2 = Q @ (K.transpose(-1, -2) @ V)
    end.record()
    torch.cuda.synchronize()
    # print(f"mathmul d2: {start.elapsed_time(end)} ms, Peak memory: {torch.cuda.max_memory_allocated()} bytes")
    peak_memory.append(torch.cuda.max_memory_allocated())
    times.append(start.elapsed_time(end))
    del result_direct2
    torch.cuda.empty_cache()

    # Third operation (S2)
    start.record()
    torch.cuda.reset_peak_memory_stats()
    result_direct3 = torch.einsum("nsq,nqw->nsw", torch.einsum("nse,nqe->nsq", Q, K), V)
    end.record()
    torch.cuda.synchronize()
    # print(f"einsum (2 steps) S2: {start.elapsed_time(end)} ms, Peak memory: {torch.cuda.max_memory_allocated()} bytes")
    peak_memory.append(torch.cuda.max_memory_allocated())
    times.append(start.elapsed_time(end))
    del result_direct3
    torch.cuda.empty_cache()
    
    # Fourth operation (d2)
    start.record()
    torch.cuda.reset_peak_memory_stats()
    result_direct4 = torch.einsum("nsw,nwe->nse", Q, torch.einsum("nse,nsw->new", K, V))
    end.record()
    torch.cuda.synchronize()
    # print(f"einsum (2 steps) d2: {start.elapsed_time(end)} ms, Peak memory: {torch.cuda.max_memory_allocated()} bytes")
    peak_memory.append(torch.cuda.max_memory_allocated())
    times.append(start.elapsed_time(end))
    del result_direct4
    torch.cuda.empty_cache()

    # Fifth operation
    start.record()
    torch.cuda.reset_peak_memory_stats()
    result_direct5 = torch.einsum("nse,nqe,nqw->nsw", Q, K, V)
    end.record()
    torch.cuda.synchronize()
    # print(f"einsum (1 step): {start.elapsed_time(end)} ms, Peak memory: {torch.cuda.max_memory_allocated()} bytes")
    peak_memory.append(torch.cuda.max_memory_allocated())
    times.append(start.elapsed_time(end))
    del result_direct5
    torch.cuda.empty_cache()
    
    
    # Sixth operation
    start.record()
    torch.cuda.reset_peak_memory_stats()
    if S < E:
        result_direct6 = (Q @ K.transpose(-1, -2)) @ V
    else:
        result_direct6 = Q @ (K.transpose(-1, -2) @ V)
    end.record()
    torch.cuda.synchronize()
    # print(f"einsum (1 step): {start.elapsed_time(end)} ms, Peak memory: {torch.cuda.max_memory_allocated()} bytes")
    peak_memory.append(torch.cuda.max_memory_allocated())
    times.append(start.elapsed_time(end))
    del result_direct6
    torch.cuda.empty_cache()
    
    # Seventh operation
    start.record()
    torch.cuda.reset_peak_memory_stats()
    if S < E:
        result_direct7 = torch.einsum("nsq,nqw->nsw", torch.einsum("nse,nqe->nsq", Q, K), V)
    else:
        result_direct7 = torch.einsum("nsw,nwe->nse", Q, torch.einsum("nse,nsw->new", K, V))
    end.record()
    torch.cuda.synchronize()
    # print(f"einsum (1 step): {start.elapsed_time(end)} ms, Peak memory: {torch.cuda.max_memory_allocated()} bytes")
    peak_memory.append(torch.cuda.max_memory_allocated())
    times.append(start.elapsed_time(end))
    del result_direct7
    torch.cuda.empty_cache()

    # assert torch.allclose(result_direct, result_direct3)
    # assert torch.allclose(result_direct2, result_direct4)
    # if S < E:
    #     assert torch.allclose(result_direct, result_direct5)
    # else:
    #     assert torch.allclose(result_direct2, result_direct5)
        
    return times, peak_memory
    



# Initialize arrays to store data
Es = np.arange(32, 2049, 16) # E values from 32 to 2048
times_all_methods = []
memory_all_methods = []

# Loop over different values of E
for E in tqdm(Es):
    times, memory = time_this(S=512, E=E)
    times_all_methods.append(times)
    memory_all_methods.append(memory)

# Convert to numpy array for easy slicing
times_all_methods = np.array(times_all_methods)
memory_all_methods = np.array(memory_all_methods)

# Plotting (times)
plt.figure(figsize=(10, 6))
for i in range(times_all_methods.shape[1]):
    plt.plot(Es, times_all_methods[:, i], label=f'Method {i+1}')
plt.xlabel('E')
plt.ylabel('Time (ms)')
plt.title('E vs. Time for Different Methods (S=512)')
plt.legend()
plt.grid()
plt.savefig('BERT_Trainer/E_vs_time.png')
plt.show()

# Plotting (memory)
plt.figure(figsize=(10, 6))
for i in range(memory_all_methods.shape[1]):
    plt.plot(Es, memory_all_methods[:, i], label=f'Method {i+1}')
plt.xlabel('E')
plt.ylabel('Memory (bytes)')
plt.title('E vs. Memory for Different Methods (S=512)')
plt.legend()
plt.grid()
plt.savefig('BERT_Trainer/E_vs_memory.png')
plt.show()










# Initialize arrays to store data
Ss = np.arange(32, 2049, 16) # S values from 32 to 2048
times_all_methods = []
memory_all_methods = []

# Loop over different values of S
for S in tqdm(Ss):
    times, memory = time_this(S=S, E=512)
    times_all_methods.append(times)
    memory_all_methods.append(memory)

# Convert to numpy array for easy slicing
times_all_methods = np.array(times_all_methods)
memory_all_methods = np.array(memory_all_methods)

# Plotting (times)
plt.figure(figsize=(10, 6))
for i in range(times_all_methods.shape[1]):
    plt.plot(Es, times_all_methods[:, i], label=f'Method {i+1}')
plt.xlabel('S')
plt.ylabel('Time (ms)')
plt.title('S vs. Time for Different Methods (E=512)')
plt.legend()
plt.grid()
plt.savefig('BERT_Trainer/S_vs_time.png')
plt.show()

# Plotting (memory)
plt.figure(figsize=(10, 6))
for i in range(memory_all_methods.shape[1]):
    plt.plot(Es, memory_all_methods[:, i], label=f'Method {i+1}')
plt.xlabel('S')
plt.ylabel('Memory (bytes)')
plt.title('S vs. Memory for Different Methods (E=512)')
plt.legend()
plt.grid()
plt.savefig('BERT_Trainer/S_vs_memory.png')
plt.show()