// Hehe
// ^w^
// UwU
// OwO
// Nyaa~ <- What I say when I'm coding in cuda. Nya~~~
// Rawr~
// >w<
// >_<
// >.<
// >:3 <- Doesn't matter if it's a cat or a dog. It's a catdog.
// >:D 
// >:P
// >:( <- This me. IDK how to code cuda kernels.
// ^w^
// UwU
// Nya~~


// ⠀⢸⠂⠀⠀⠀⠘⣧⠀⠀⣟⠛⠲⢤⡀⠀⠀⣰⠏⠀⠀⠀⠀⠀⢹⡀
// ⠀⡿⠀⠀⠀⠀⠀⠈⢷⡀⢻⡀⠀⠀⠙⢦⣰⠏⠀⠀⠀⠀⠀⠀⢸⠀
// ⠀⡇⠀⠀⠀⠀⠀⠀⢀⣻⠞⠛⠀⠀⠀⠀⠻⠀⠀⠀⠀⠀⠀⠀⢸⠀
// ⠀⡇⠀⠀⠀⠀⠀⠀⠛⠓⠒⠓⠓⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀
// ⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠀
// ⠀⢿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⣀⠀⠀⢀⡟⠀
// ⠀⠘⣇⠀⠘⣿⠋⢹⠛⣿⡇⠀⠀⠀⠀⣿⣿⡇⠀⢳⠉⠀⣠⡾⠁⠀
// ⣦⣤⣽⣆⢀⡇⠀⢸⡇⣾⡇⠀⠀⠀⠀⣿⣿⡷⠀⢸⡇⠐⠛⠛⣿⠀
// ⠹⣦⠀⠀⠸⡇⠀⠸⣿⡿⠁⢀⡀⠀⠀⠿⠿⠃⠀⢸⠇⠀⢀⡾⠁⠀
// ⠀⠈⡿⢠⢶⣡⡄⠀⠀⠀⠀⠉⠁⠀⠀⠀⠀⠀⣴⣧⠆⠀⢻⡄⠀⠀
// ⠀⢸⠃⠀⠘⠉⠀⠀⠀⠠⣄⡴⠲⠶⠴⠃⠀⠀⠀⠉⡀⠀⠀⢻⡄⠀
// ⠀⠘⠒⠒⠻⢦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣤⠞⠛⠒⠛⠋⠁⠀
// ⠀⠀⠀⠀⠀⠀⠸⣟⠓⠒⠂⠀⠀⠀⠀⠀⠈⢷⡀⠀⠀⠀⠀⠀⠀⠀
// ⠀⠀⠀⠀⠀⠀⠀⠙⣦⠀⠀⠀⠀⠀⠀⠀⠀⠈⢷⠀⠀⠀⠀⠀⠀⠀
// ⠀⠀⠀⠀⠀⠀⠀⣼⣃⡀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣆⠀⠀⠀⠀⠀⠀
// ⠀⠀⠀⠀⠀⠀⠀⠉⣹⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⠀⠀⠀⠀⠀⠀
// ⠀⠀⠀⠀⠀⠀⠀⠀⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡆⠀⠀⠀⠀⠀
// OOOH you like coding cuda kernels? You're an insane person. UwU



#include <cuda_runtime.h> // For cudaMemcpy and cudaFree
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/autocast_mode.h>
// #include <torch/extension.h>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <chrono>

#include <cuda_fp16.h> // Include CUDA half-precision definitions



// General AtomicAdd_
template<typename T>
__device__ void AtomicAdd_(T* address, T val) {
    atomicAdd(address, val);
}
// Specialization for half precision
template<>
__device__ void AtomicAdd_(at::Half* address, at::Half val) {
    atomicAdd(reinterpret_cast<__half*>(address), *reinterpret_cast<__half*>(&val));
}
// Specialization for bfloat16 half precision
template<>
__device__ void AtomicAdd_(at::BFloat16* address, at::BFloat16 val) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), *reinterpret_cast<__nv_bfloat16*>(&val));
}



// General __shfl_down_sync
template<typename T>
__device__ T __shfl_down_sync_(unsigned mask, T val, int delta, int width = warpSize) {
    return __shfl_down_sync(mask, val, delta, width);
}
// Specialization for half precision
template<>
__device__ at::Half __shfl_down_sync_(unsigned mask, at::Half val, int delta, int width) {
    return __shfl_down_sync(mask, *reinterpret_cast<__half*>(&val), delta, width);
}
// Specialization for bfloat16 half precision
template<>
__device__ at::BFloat16 __shfl_down_sync_(unsigned mask, at::BFloat16 val, int delta, int width) {
    return __shfl_down_sync(mask, *reinterpret_cast<__nv_bfloat16*>(&val), delta, width);
}




__global__
void block_sum_reduce(unsigned int* const d_block_sums, 
	const unsigned int* const d_in,
	const unsigned int d_in_len)
{
	extern __shared__ unsigned int s_out[];

	unsigned int max_elems_per_block = blockDim.x * 2;
	unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	
	// Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	s_out[threadIdx.x] = 0;
	s_out[threadIdx.x + blockDim.x] = 0;

	__syncthreads();

	// Copy d_in to shared memory per block
	if (glbl_tid < d_in_len)
	{
		s_out[threadIdx.x] = d_in[glbl_tid];
		if (glbl_tid + blockDim.x < d_in_len)
			s_out[threadIdx.x + blockDim.x] = d_in[glbl_tid + blockDim.x];
	}
	__syncthreads();

	// Actually do the reduction
	for (unsigned int s = blockDim.x; s > 0; s >>= 1) {
		if (tid < s) {
			s_out[tid] += s_out[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		d_block_sums[blockIdx.x] = s_out[0];
}


// template <typename T>
// __device__ T blockReduce(T* smem,  // Pointer to shared memory
//             T defaultVal) // Default value - used to fill the shared memory
// {
//   __syncthreads();

//   T warpVal = defaultVal;

//   // First warp will perform per-warp reductions for the remaining warps
//   if (threadIdx.x < C10_WARP_SIZE) {
//     int lane = threadIdx.x % C10_WARP_SIZE;
//     if (lane < blockDim.x / C10_WARP_SIZE) {
// #pragma unroll
//       for (int i = 0; i < C10_WARP_SIZE; ++i) {
//         warpVal = warpVal + smem[lane * C10_WARP_SIZE + i];
//       }
//       smem[lane] = warpVal;
//     }
//   }

//   __syncthreads();

//   // First thread will perform a reduction of the above per-warp reductions
//   T blockVal = defaultVal;

//   if (threadIdx.x == 0) {
//     for (int i = 0; i < blockDim.x / C10_WARP_SIZE; ++i) {
//       blockVal = blockVal + smem[i];
//     }
//     smem[0] = blockVal;
//   }

//   // Sync and broadcast
//   __syncthreads();
//   return smem[0];
// }



template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync_(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduce(T val) {
    static __shared__ T shared[32]; // Assuming a maximum of 32 warps per block
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Reduce within each warp
    val = warpReduceSum(val);

    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = val;

    __syncthreads();

    // Ensure we only proceed with the first warp for final reduction
    if (threadIdx.x < blockDim.x / warpSize) val = shared[lane];else val = 0;
    if (wid == 0) val = warpReduceSum(val); // Final reduce within the first warp

    return val;
}



template<typename T, int sum_size1, int sum_size2>
__global__ void forward_kernel(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    const int block_size) {
    
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d_v = blockIdx.z; // Dimension index within d_V
    int d_k = threadIdx.x; // Dimension index within d_k


    // Ensure we are within bounds
    if (d_k >= d_K+1 || d_v >= d_V) {
        return;
    }


    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);

    // Initialize the shared memory to 0
    if (d_k < d_K) {
        shared_memory[d_k] = 0;
    }



    // Iterate over the entire sequence
    for (int s = 0; s < S; s++) {
        // Wait for all threads to finish the previous iteration
        __syncthreads();

        if (d_k < d_K) {
            // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_k]
            shared_memory[d_k] += V[((n * H + h) * S + s) * d_V + d_v] * K[((n * H + h) * S + s) * d_K + d_k];
            __syncthreads();

            // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
            shared_memory[d_K + d_k] = shared_memory[d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
        }

        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        if (d_k == 0) {
            T sum_ = 0;
            for (int i = 0; i < d_K; i++) {
                sum_ += shared_memory[d_K + i];
            }
            output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        }
    }

    // Wait for all threads to finish the previous iteration
    __syncthreads();

    return;



    __syncthreads();
    // Iterate over the entire sequence
    for (int s = 0; s < S; s++) {
        // Wait for all threads to finish the previous iteration
        // __syncthreads();

        // If the thread is 0, cache the value of V in shared memory
        if (d_k == 0) {
            shared_memory[d_K*2] = V[((n * H + h) * S + s) * d_V + d_v];
        }
        __syncthreads();

        // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_k]
        shared_memory[d_k] += shared_memory[d_K*2] * K[((n * H + h) * S + s) * d_K + d_k];

        // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
        shared_memory[d_K + d_k] = shared_memory[d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
        
        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        T out = blockReduce<T>(shared_memory[d_K + d_k]);
        if (d_k == 1) {
            output[((n * H + h) * S + s) * d_V + d_v] = out;
        }
        // output[((n * H + h) * S + s) * d_V + d_v] = blockReduce<T>(shared_memory[d_K + d_k]);
        continue;



        /* method 1
        // Load data into register
        T val = shared_memory[d_K + d_k];

        // Parallel reduction
        for (unsigned int stride = d_K / 2; stride > 0; stride >>= 1) {
            __syncthreads(); // Synchronize threads in the block
            if (d_k < stride) {
                val += shared_memory[d_K + d_k + stride];
                shared_memory[d_K + d_k] = val; // Store back if needed, or use another strategy to keep track of the sum
            }
        }
        __syncthreads(); // Synchronize threads in the block
        if (d_k == 0) {
            output[((n * H + h) * S + s) * d_V + d_v] = val;
        }
        */





        // Assume d_k, d_K, shared_memory, and output are defined similarly to the original code.
        T val = shared_memory[d_K + d_k];
        for (unsigned int stride = d_K / 2; stride > 32; stride >>= 1) { // Assuming warp size of 32
            __syncthreads();
            if (d_k < stride) {
                val += shared_memory[d_K + d_k + stride];
            }
        }

        // Only synchronize before going into warp-specific operations
        __syncthreads();

        // Handle the final warp without synchronization, using warp primitives if possible
        if (d_k < 32) {
            // Assuming warp size is 32, manually unroll the last steps of the reduction
            // Note: This part might require adjustments based on the actual warp size and whether warp shuffle functions are used.
            if (d_K / 2 > 32) shared_memory[d_K + d_k] = val;
            if (d_k < 16) shared_memory[d_K + d_k] += shared_memory[d_K + d_k + 16];
            if (d_k < 8) shared_memory[d_K + d_k] += shared_memory[d_K + d_k + 8];
            if (d_k < 4) shared_memory[d_K + d_k] += shared_memory[d_K + d_k + 4];
            if (d_k < 2) shared_memory[d_K + d_k] += shared_memory[d_K + d_k + 2];
            if (d_k < 1) shared_memory[d_K + d_k] += shared_memory[d_K + d_k + 1];
        }

        // Final output
        if (d_k == 0) {
            output[((n * H + h) * S + s) * d_V + d_v] = shared_memory[d_K];
        }



        continue;

        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        for (unsigned int s=d_K/2; s>0; s>>=1) {
            if (d_k < s) {
                shared_memory[d_K + d_k] += shared_memory[d_K + d_k + s];
            }
            __syncthreads();
        }
        // write result for this block to global mem
        if (d_k == 0) {
            output[((n * H + h) * S + s) * d_V + d_v] = shared_memory[d_K];
        }
        continue;

        // break into d_K/sum_size1 blocks
        // Every thread with an index of a factor of sum_size1 sums sum_size1 elements above its index
        // into its own index in the second half of the shared memory
        if (d_k % sum_size1 == 0) {
            for (int i = d_k+1; i < min(d_k + sum_size1, d_K); i++) {
                shared_memory[d_K + d_k] += shared_memory[d_K + i];
            }
        }

        // break into d_K/sum_size1*sum_size2 blocks
        // Every thread with an index of a factor of sum_size1*sum_size2 the sum_size2 threads above its index
        // into its own index in the second half of the shared memory
        if (d_k % (sum_size1*sum_size2) == 0) {
            for (int i = d_k+sum_size1; i < min(d_k + sum_size1*sum_size2, d_K); i+=sum_size1) {
                shared_memory[d_K + d_k] += shared_memory[d_K + i];
            }
        }

        // The 0th thread sums all the threads with an index of factor sum_size1*sum_size2
        // into its own index in the second half of the shared memory, then into the global memory
        if (d_k == 0) {
            T sum_ = 0;
            for (int i = 0; i < d_K; i += sum_size1*sum_size2) {
                sum_ += shared_memory[d_K + i];
            }
            output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        }
        continue;


        if (d_k == 0) {
            T sum_ = 0;
            for (int i = 0; i < d_K; i++) {
                sum_ += shared_memory[d_K + i];
            }
            output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        }

        // Alternative, atomicAdd. ALl threads will be writing to the same memory location
        // AtomicAdd_(&output[((n * H + h) * S + s) * d_V + d_v], shared_memory[d_K + d_k]);

        // Alternative: add to shared memory (d_k + 1) and then write to global memory
        if (d_k != d_K) {
            
        }
        __syncthreads();
        if (d_k == d_K) {
            
        }
    }
}



// Wrapper function to orchestrate the computation
template<typename T>
void forward_call(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;

    dim3 grid(N, H, d_V);
    dim3 block(d_K);
    forward_kernel<T, 8, 4><<<grid, block, 2*d_K*sizeof(T), stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size);

    // Device synchronization is needed to ensure the kernel is complete before the output is used
    cudaDeviceSynchronize();
}
















































template<typename T>
__global__ void forward_kernel_(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    const int block_size) {
    
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d_v = threadIdx.x; // Dimension index within d_V


    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);












    // Allocate memory for an array of size `d` inside the kernel
    T* mem = (T*)malloc(d_K * sizeof(T));
    if (mem == NULL) return;
    // Example computation - initialize the allocated array
    for (int i = 0; i < d_K; ++i) {
        mem[i] = 0;
    }


    // Memcopy Q, K, and V to shared memory. Each thread memcopies all S elements for a given n, h, and d_v
    cudaMemcpy(shared_memory, K, d_K * sizeof(T), cudaMemcpyDeviceToDevice);




    // Iterate over the sequence dimension
    for (int s = 0; s < S; s++) {
        T out = 0;

        // Iterate over the d_k dimension
        for (int d_k = 0; d_k < d_K; d_k++) {
            // 1: Compute V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to the dth element of the memory
            mem[d_k] += V[((n * H + h) * S + s) * d_V + d_v] * K[((n * H + h) * S + s) * d_K + d_k];

            // 2: Multiply the dth element of the memory by Q[:, :, s, d_k] and add it to out
            out += mem[d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
        }

        // Store the result in the output tensor
        output[((n * H + h) * S + s) * d_V + d_v] = out;
    }

    // Free the memory
    free(mem);












    return;




    /*





    // // Iterate over the entire sequence
    // for (int s = 0; s < S; s++) {
    //     // Wait for all threads to finish the previous iteration
    //     // __syncthreads();

    //     if (d_k < d_K) {
    //         // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_k]
    //         shared_memory[d_k] += V[((n * H + h) * S + s) * d_V + d_v] * K[((n * H + h) * S + s) * d_K + d_k];
    //         __syncthreads();

    //         // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
    //         shared_memory[d_K + d_k] = shared_memory[d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
    //     }

    //     // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
    //     __syncthreads();
    //     if (d_k == d_K) {
    //         T sum_ = 0;
    //         for (int i = 0; i < d_K; i++) {
    //             sum_ += shared_memory[d_K + i];
    //         }
    //         output[((n * H + h) * S + s) * d_V + d_v] = sum_;
    //     }
    // }



    __syncthreads();
    // Iterate over the entire sequence
    for (int s = 0; s < S; s++) {
        // Wait for all threads to finish the previous iteration
        // __syncthreads();

        // If the thread is 0, cache the value of V in shared memory
        if (d_k == 0) {
            shared_memory[d_K*2] = V[((n * H + h) * S + s) * d_V + d_v];
        }
        __syncthreads();

        // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_k]
        shared_memory[d_k] += shared_memory[d_K*2] * K[((n * H + h) * S + s) * d_K + d_k];

        // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
        shared_memory[d_K + d_k] = shared_memory[d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
        

        
        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        if (d_k == 0) {
            T sum_ = 0;
            for (int i = 0; i < d_K; i++) {
                sum_ = shared_memory[d_K + i];
            }
            output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        }
        continue;



        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        // method 1
        // Load data into register
        T val = shared_memory[d_K + d_k];

        // Parallel reduction
        for (unsigned int stride = d_K / 2; stride > 0; stride >>= 1) {
            __syncthreads(); // Synchronize threads in the block
            if (d_k < stride) {
                val += shared_memory[d_K + d_k + stride];
                shared_memory[d_K + d_k] = val; // Store back if needed, or use another strategy to keep track of the sum
            }
        }
        __syncthreads(); // Synchronize threads in the block
        if (d_k == 0) {
            output[((n * H + h) * S + s) * d_V + d_v] = val;
        }
    }


    */
}


// Wrapper function to orchestrate the computation
template<typename T>
void forward_call_(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;

    dim3 grid(N, H);
    dim3 block(d_V);
    forward_kernel_<T><<<grid, block, 0, stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size);
}





























template<typename T>
__global__ void efficientMatrixMultiply(const T* A, const T* B, const T* C, T* output, int N, int H, int S, int d) {
    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T* shared_memory = reinterpret_cast<T *>(shared_memory_uchar);

    int n = blockIdx.z / H;
    int h = blockIdx.z % H;

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.x * blockDim.x + tx;
    int col = blockIdx.y * blockDim.y + ty;
    
    if (row < d && col < d) {
        T sum = 0.0;
        for (int k = 0; k < S; ++k) {
            sum += B[(n * H + h) * S + k * d + row] * C[(n * H + h) * S + k * d + col]; // Compute B^T x C
        }
        shared_memory[row * d + col] = sum;
    }
    __syncthreads();

    // Now multiply A x (B^T x C), where (B^T x C) is stored in 'shared_memory'
    if (row < S && col < d) {
        T finalSum = 0.0;
        for (int i = 0; i < d; ++i) {
            finalSum += A[(n * H + h) * S + row * d + i] * shared_memory[i * d + col];
        }
        output[(n * H + h) * S + row * d + col] = finalSum;
    }
}

template<typename T>
__global__ void causalMatrixMultiply(const T* A, const T* B, const T* C, T* output, int N, int H, int S, int d) {
    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T* shared_memory = reinterpret_cast<T *>(shared_memory_uchar);

    int n = blockIdx.z / H;
    int h = blockIdx.z % H;

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.x * blockDim.x + tx;
    int col = blockIdx.y * blockDim.y + ty;

    if (row < S && col < d) {
        T result = 0.0;
        for (int i = 0; i <= row; ++i) { // Causality: only use up to the current row
            T BtC = 0.0;
            for (int j = 0; j < d; ++j) {
                BtC += B[(n * H + h) * S + i * d + j] * C[(n * H + h) * S + i * d + j]; // Computing B^T x C causally
            }
            result += A[(n * H + h) * S + row * d + i] * BtC;
        }
        output[(n * H + h) * S + row * d + col] = result;
    }
}

template<typename T>
void forward_call__(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;
    int d = D;

    dim3 blockSize(32, 32); // Or another configuration fitting your GPU
    dim3 gridSize((S + blockSize.x - 1) / blockSize.x, (d + blockSize.y - 1) / blockSize.y, N * H);
    size_t sharedMemSize = d * d * sizeof(T); // Adjust based on actual needs

    causalMatrixMultiply<<<gridSize, blockSize, sharedMemSize>>>(Q, K, V, output, N, H, S, D);

    // cudaDeviceSynchronize();


//     dim3 grid(N, H);
//     dim3 block(d_V);
//     forward_kernel__<T><<<grid, block, 0, stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size);
}



























































































// C++ interface
template<typename dtype_>
torch::Tensor forward_(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V, const int8_t block_size, bool inplace = false) {
    // Check tensor requirements, e.g., dtype, device, etc.
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");

    // Get tensor dimensions
    int N = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    // Get the data type, could be auto casted
    auto data_type = at::autocast::is_enabled() && Q.scalar_type() == at::kFloat ? at::kHalf : Q.scalar_type();

    // // Unsqueeze K along the last dimension and V along the second-to-last dimension
    // auto K = K_orig.unsqueeze(-1); // (N, H, S, D, 1)
    // auto V = V_orig.unsqueeze(-2); // (N, H, S, 1, D)
    // Unsqueeze not needed as I am making the kernel hehe UwU

    // Ensure the tensors are contiguous
    Q = Q.contiguous().to(data_type);
    K = K.contiguous().to(data_type);
    V = V.contiguous().to(data_type);

    // Create the output tensor
    torch::Tensor output;
    if (inplace) {
        output = V;
    } else {
        output = torch::zeros({N, H, S, D}, torch::TensorOptions().dtype(data_type).device(Q.device()));
    }

    // https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)Q.get_device()};

    // Call the CUDA kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "forward_cuda", ([&] {
        forward_call<scalar_t>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, H, S, D, block_size);
    }));

    return output;
}

// TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
//     m.def("float32", forward_<float>);
//     m.def("float16", forward_<at::Half>);
//     try {
//         m.def("bfloat16", forward_<at::BFloat16>);
//     } catch (const std::exception& e) {
//         std::cout << "GPU does not support bfloat16. Skipping..." << std::endl;
//         // std::cerr << "Error: " << e.what() << std::endl;
//     }
// }

TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, Autocast, m) {
    m.impl("float32", forward_<float>);
    m.impl("float64", forward_<double>);
    m.impl("float16", forward_<at::Half>);
    try {
        m.impl("bfloat16", forward_<at::BFloat16>);
    } catch (const std::exception& e) {
        std::cout << "GPU does not support bfloat16. Skipping..." << std::endl;
        // std::cerr << "Error: " << e.what() << std::endl;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float32", &forward_<float>);
    m.def("float64", &forward_<double>);
    m.def("float16", &forward_<at::Half>);
    try {
        m.def("bfloat16", &forward_<at::BFloat16>);
    } catch (const std::exception& e) {
        std::cout << "GPU does not support bfloat16. Skipping..." << std::endl;
        // std::cerr << "Error: " << e.what() << std::endl;
    }
}



// // Debugging
// #include <iostream>
// #include <chrono>
// // dummy main function
// int main() {
//     // Set the device
//     torch::Device device(torch::kCUDA, 0);

//     // Set the tensor dimensions
//     int N = 16;
//     int H = 8;
//     int S = 64;
//     int D = 32;

//     // Create input tensors
//     auto Q = torch::rand({N, H, S, D}, device);
//     auto K = torch::rand({N, H, S, D}, device);
//     auto V = torch::rand({N, H, S, D}, device);

//     // Create output tensor
//     auto output = torch::zeros({N, H, S, D}, device);

//     // Call the custom CUDA kernel
//     auto start = std::chrono::high_resolution_clock::now();
//     compute_and_contract_call(Q, K, V, output, 5);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "Elapsed time: " << elapsed.count() << " s\n";

//     return 0;
// }




// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⡿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⡟⠀⣠⣀⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣄⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⡟⠀⣼⣿⣿⣿⣦⣄⠙⠻⣿⣿⣿⣿⣿⣿⣿⠀⢻⣷⣦⣈⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⠃⢰⣿⣿⣿⣿⣿⣿⣿⣦⡍⠙⠉⣁⣠⣤⣤⣄⡀⢻⣿⣿⣿⣦⣄⣈⠙⠿⢿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⣀⣠⣴⣶⣿⣷⡄⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⡏⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣄⣀⠛⢿⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣿⣿⣷⣶⣄⠛⣿⣿⣿⡿⠟⠋⣠⣴⣾⣿⣿⣿⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⡇⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣌⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⠘⣿⠋⠀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⠁⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⡄⠉⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⣦⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢻⣿⡀⢻⣿⣿⣿⠏⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣷⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡘⣿⠃⣸⣿⣿⠏⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⡿⠿⠿⠛⠃⣠⣿⣿⡿⠟⠁⢀⣀⣀⡀⠉⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣿⡿⠋⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣷⡈⢶⣶⣿⣿⣿⣿⣦⣤⣾⣿⣿⣿⣿⣷⣀⢘⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠉⠀⣀⣀⣀⠀⠉⠻⣿⣿⣿⣿⣿⣿⠟⠀⠀⠛⠛⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣷⣄⡛⠟⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⣠⣶⣿⣿⣿⣿⣿⣷⣄⠈⣿⣿⣿⣿⣿⣶⣾⣿⡟⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⡟⢁⣾⡟⠿⠛⠉⢻⣿⣿⣿⣿⣧⣀⡀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠁⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⡿⠀⣿⣿⣿⣿⣿⡁⣉⣁⣤⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⢿⡿⠿⢿⣿⣿⡀⠠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⡟⠀⠚⠛⣉⣉⣉⡉⠛⢿⣿⣿⣿⣿⣿⣿⡿⢿⣿⠿⢿⣿⣿⡏⣿⣿⣿⣿⣿⣧⣴⣶⣧⡀⢉⣠⣶⣿⣿⣿⣷⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣷⣶⣿⣿⣿⣿⣿⣷⣦⡀⠙⠻⢿⣿⣿⣿⣧⣌⠉⣠⣬⣍⠋⢁⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⣿⠿⢿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⣉⠙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠟⠛⠋⠉⠠⠤⣤⣴⣶⣦⣤⣤⣄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠠⣤⣤⣤⣤⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⣿⣿⣿⠃⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠉⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⣉⣠⣤⣶⣶⣤⣤⣄⠀⠸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⠛⢁⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⡈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣴⣿⣿⣿⣿⣿⣿⣿⡟⢁⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣀⣠⣤⡄⠸⣿⣿⣿⣿⣇⠸⣿⡏⢹⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⣿⣿⣿⣿⣿⣦⣈⠁⠘⠿⣿⡇⢸⠿⠟⢉⣠⣿⣿⣿⣿⣿⣷⡀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⣧⡄⠹⣿⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⣤⣤⣤⡆⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠹⣿⠃⢰⣿⣷⣄⠘⣿⣿⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⣰⡀⠈⠀⣿⣿⣿⣿⣄⠈⢻⣿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⡏⢻⣿⣿⣿⣿⣇⠹⣿⣿⣿⣿⣿⣿⣿⡿⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⣰⣿⣇⠀⢰⣿⣿⣿⣿⣿⣇⠈⢿⣿⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣧⠘⣿⣿⣿⣿⣿⣄⠻⣿⣿⣿⣿⡿⠟⢀⠰⠻⠿⠿⣿⣿⣿⣿⣿⣿⣿⡟⢀⣼⣿⣿⣿⢠⣿⣿⣿⣿⣿⣿⣿⣇⠈⢻⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠀⣿⣿⣿⣿⣿⣿⣿⣿⡄⢹⣿⣿⣿⣿⣿⣶⣤⣤⣤⣤⣴⣾⣿⣶⡶⠂⣴⣿⣿⣿⣿⡿⠟⠉⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠈⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⢀⣰⣶⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣀⠘⠿⢿⠿⠛⠁⣀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠀⣿⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣠⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⢁⣠⣤⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿⣿⣿⣿⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣼⣿
// ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢼⣿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠁⣴⣿⣿