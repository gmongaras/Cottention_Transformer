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
    // if (threadIdx.x < blockDim.x / warpSize) val = shared[lane];else val = 0;
    // The ? operator does not have divergence
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : (T)0;
    if (wid == 0) val = warpReduceSum(val); // Final reduce within the first warp

    __syncthreads();

    return val;
}




























template<typename T>
__global__ void forward_kernel(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    const int block_size) {
    
    int n = blockIdx.y; // Batch index
    int h = blockIdx.z; // Head index
    int d_v = blockIdx.x; // Dimension index within d_V
    int d_k = threadIdx.x; // Dimension index within d_k


    // // Ensure we are within bounds
    // if (d_k >= d_K || d_v >= d_V) {
    //     return;
    // }


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
        // __syncthreads();

        if (d_k < d_K) {
            // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_k]
            shared_memory[d_k] += V[((n * H + h) * S + s) * d_V + d_v] * K[((n * H + h) * S + s) * d_K + d_k];
            __syncthreads();

            // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
            shared_memory[d_K + d_k] = shared_memory[d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
        }

        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        // if (d_k == 0) {
        //     T sum_ = 0;
        //     for (int i = 0; i < d_K; i++) {
        //         sum_ += shared_memory[d_K + i];
        //     }
        //     output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        // }

        T out = blockReduce<T>(shared_memory[d_K + d_k]);
        if (d_k == 0) {
            output[((n * H + h) * S + s) * d_V + d_v] = out;
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

    // What is the maximum number of threads per block?
    int max_threads_per_block = 1024;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_threads_per_block, forward_kernel<T>, block_size, 0);

    // Get the number of blocks we can do in parallel - the number of times the dimension divides the number of threads
    int num_blocks = floor(max_threads_per_block / d_V);
    num_blocks = 1;

    // dim3 grid((int)(d_V/num_blocks), N, H); // Note that d_V is first as it is the fastest changing dimension
    // dim3 block(d_K, num_blocks); // Note that d_K is first as it is the fastest changing dimension
    // dim3 block(max_threads_per_block);
    dim3 grid(d_V, N, H);
    dim3 block(d_K);
    forward_kernel<T><<<grid, block, 2*d_K*sizeof(T), stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size);
}


















template<typename T>
__global__ void forward_kernel_double_over_d(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    const int block_size) {
    
    int n = blockIdx.y; // Batch index
    int h = blockIdx.z; // Head index
    int d_v = blockIdx.x; // Dimension index within d_V
    int d_k = threadIdx.x*2; // Dimension index within d_k


    // // Ensure we are within bounds
    // if (d_k >= d_K || d_v >= d_V) {
    //     return;
    // }


    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);

    // Initialize the shared memory to 0
    if (d_k < d_K) {
        shared_memory[d_k] = shared_memory[d_k + 1] = 0;
    }



    // Iterate over the entire sequence
    for (int s = 0; s < S; s++) {
        // Wait for all threads to finish the previous iteration
        // __syncthreads();

        if (d_k < d_K) {
            // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_k]
            shared_memory[d_k] += V[((n * H + h) * S + s) * d_V + d_v] * K[((n * H + h) * S + s) * d_K + d_k];
            shared_memory[d_k+1] += V[((n * H + h) * S + s) * d_V + d_v] * K[((n * H + h) * S + s) * d_K + d_k + 1];
            __syncthreads();

            // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
            shared_memory[d_K + d_k] = shared_memory[d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
            shared_memory[d_K + d_k + 1] = shared_memory[d_k + 1] * Q[((n * H + h) * S + s) * d_K + d_k + 1];
        }

        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        // if (d_k == 0) {
        //     T sum_ = 0;
        //     for (int i = 0; i < d_K; i++) {
        //         sum_ += shared_memory[d_K + i];
        //     }
        //     output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        // }

        shared_memory[d_K + d_k] += shared_memory[d_K + d_k + 1];
        T out = blockReduce<T>(shared_memory[d_K + d_k]);
        if (d_k == 0) {
            output[((n * H + h) * S + s) * d_V + d_v] = out;
        }
    }
}



// Wrapper function to orchestrate the computation
template<typename T>
void forward_call_double_over_d(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;

    // What is the maximum number of threads per block?
    int max_threads_per_block = 1024;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_threads_per_block, forward_kernel<T>, block_size, 0);

    // Get the number of blocks we can do in parallel - the number of times the dimension divides the number of threads
    int num_blocks = floor(max_threads_per_block / d_V);
    num_blocks = 1;

    // dim3 grid((int)(d_V/num_blocks), N, H); // Note that d_V is first as it is the fastest changing dimension
    // dim3 block(d_K, num_blocks); // Note that d_K is first as it is the fastest changing dimension
    // dim3 block(max_threads_per_block);
    dim3 grid(d_V, N, H);
    dim3 block((int)d_K/2);
    forward_kernel_double_over_d<T><<<grid, block, 2*d_K*sizeof(T), stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size);
}

























template<typename T>
__global__ void forward_kernel_double_over_d__v_cache(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    const int block_size) {
    
    int n = blockIdx.y; // Batch index
    int h = blockIdx.z; // Head index
    int d_v = blockIdx.x; // Dimension index within d_V
    int d_k = threadIdx.x*2; // Dimension index within d_k

    int shared_memory_row_size = 2 * d_K;


    // // Ensure we are within bounds
    // if (d_k >= d_K || d_v >= d_V) {
    //     return;
    // }


    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);

    // Initialize the shared memory to 0
    if (d_k < d_K) {
        shared_memory[d_k] = shared_memory[d_k + 1] = 0;
    }



    // Cache the V values
    for (int s = d_k; s < S; s += d_K) {
        shared_memory[shared_memory_row_size + s] = V[((n * H + h) * S + s) * d_V + d_v];
        if (s + 1 < S)
            shared_memory[shared_memory_row_size + s + 1] = V[((n * H + h) * S + s + 1) * d_V + d_v];
    }
    __syncthreads();



    // Iterate over the entire sequence
    for (int s = 0; s < S; s++) {
        // Wait for all threads to finish the previous iteration
        // __syncthreads();

        if (d_k < d_K) {
            T v = shared_memory[shared_memory_row_size + s];

            // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_k]
            shared_memory[d_k] += v * K[((n * H + h) * S + s) * d_K + d_k];
            shared_memory[d_k + 1] += v * K[((n * H + h) * S + s) * d_K + d_k + 1];
            __syncthreads();

            // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
            shared_memory[d_K + d_k] = shared_memory[d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
            shared_memory[d_K + d_k + 1] = shared_memory[d_k + 1] * Q[((n * H + h) * S + s) * d_K + d_k + 1];
        }

        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        // if (d_k == 0) {
        //     T sum_ = 0;
        //     for (int i = 0; i < d_K; i++) {
        //         sum_ += shared_memory[d_K + i];
        //     }
        //     output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        // }

        T tmp = shared_memory[d_K + d_k] + shared_memory[d_K + d_k + 1];
        T out = blockReduce<T>(tmp);
        if (d_k == 0) {
            output[((n * H + h) * S + s) * d_V + d_v] = out;
        }
    }
}



// Wrapper function to orchestrate the computation
template<typename T>
void forward_call_double_over_d__v_cache(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;

    // dim3 grid((int)(d_V/num_blocks), N, H); // Note that d_V is first as it is the fastest changing dimension
    // dim3 block(d_K, num_blocks); // Note that d_K is first as it is the fastest changing dimension
    // dim3 block(max_threads_per_block);
    dim3 grid(d_V, N, H);
    dim3 block((int)d_K/2);
    forward_kernel_double_over_d__v_cache<T><<<grid, block, 2*d_K*sizeof(T) + S*sizeof(T), stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size);
}



























template<typename T>
__global__ void forward_kernel_multiple_over_d__v_cache(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    const int block_size, const int dim_skip) {
    
    int n = blockIdx.y; // Batch index
    int h = blockIdx.z; // Head index
    int d_v = blockIdx.x; // Dimension index within d_V
    int d_k = threadIdx.x*dim_skip; // Dimension index within d_k

    int shared_memory_row_size = 2 * d_K;


    // // Ensure we are within bounds
    // if (d_k >= d_K || d_v >= d_V) {
    //     return;
    // }


    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);

    // Initialize the shared memory to 0
    if (d_k < d_K) {
        for (int i = 0; i < dim_skip; i++) {
            shared_memory[d_k + i]  = 0;
        }
    }



    // Cache the V values
    for (int s = d_k; s < S; s += d_K) {
        for (int i = 0; i < dim_skip; i++) {
            if (s + i < S)
                shared_memory[shared_memory_row_size + s + i] = V[((n * H + h) * S + s + i) * d_V + d_v];
        }
    }
    __syncthreads();



    // Iterate over the entire sequence
    for (int s = 0; s < S; s++) {
        // Wait for all threads to finish the previous iteration
        // __syncthreads();

        if (d_k < d_K) {
            T v = shared_memory[shared_memory_row_size + s];

            // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_k]
            for (int i = 0; i < dim_skip; i++) {
                if (d_k + i < d_K)
                    shared_memory[d_k + i] += v * K[((n * H + h) * S + s) * d_K + d_k + i];
            }
            __syncthreads();

            // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
            for (int i = 0; i < dim_skip; i++) {
                if (d_k + i < d_K)
                    shared_memory[d_K + d_k + i] = shared_memory[d_k + i] * Q[((n * H + h) * S + s) * d_K + d_k + i];
            }
        }

        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        // if (d_k == 0) {
        //     T sum_ = 0;
        //     for (int i = 0; i < d_K; i++) {
        //         sum_ += shared_memory[d_K + i];
        //     }
        //     output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        // }

        T tmp = 0;
        for (int i = 0; i < dim_skip; i++) {
            if (d_k + i < d_K)
                tmp += shared_memory[d_K + d_k + i];
        }
        T out = blockReduce<T>(tmp);
        if (d_k == 0) {
            output[((n * H + h) * S + s) * d_V + d_v] = out;
        }
    }
}



// Wrapper function to orchestrate the computation
template<typename T>
void forward_call_multiple_over_d__v_cache(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;

    int dim_skip = 2;

    // dim3 grid((int)(d_V/num_blocks), N, H); // Note that d_V is first as it is the fastest changing dimension
    // dim3 block(d_K, num_blocks); // Note that d_K is first as it is the fastest changing dimension
    // dim3 block(max_threads_per_block);
    dim3 grid(d_V, N, H);
    dim3 block(ceil((float)d_K/(float)dim_skip));
    forward_kernel_multiple_over_d__v_cache<T><<<grid, block, 2*d_K*sizeof(T) + S*sizeof(T), stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size, dim_skip);
}


















































template<typename T>
__global__ void forward_kernel_double_over_s(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    const int block_size) {
    
    int n = blockIdx.y; // Batch index
    int h = blockIdx.z; // Head index
    int d_v = blockIdx.x; // Dimension index within d_V
    int d_k = threadIdx.x; // Dimension index within d_k

    int shared_memory_row_size = 2 * d_K;


    // // Ensure we are within bounds
    // if (d_k >= d_K || d_v >= d_V) {
    //     return;
    // }


    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);

    // Initialize the shared memory to 0
    if (d_k < d_K) {
        shared_memory[d_k] = shared_memory[shared_memory_row_size + d_k] = 0;
    }



    // Iterate over the entire sequence
    for (int s = 0; s < S; s+=2) {
        // Wait for all threads to finish the previous iteration
        // __syncthreads();

        if (d_k < d_K) {
            // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_k]
            shared_memory[d_k] += V[((n * H + h) * S + s) * d_V + d_v] * K[((n * H + h) * S + s) * d_K + d_k];
            shared_memory[shared_memory_row_size + d_k] += V[((n * H + h) * S + s+1) * d_V + d_v] * K[((n * H + h) * S + s+1) * d_K + d_k];
            __syncthreads();

            // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
            shared_memory[d_K + d_k] = shared_memory[d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
            shared_memory[shared_memory_row_size + d_K + d_k] = shared_memory[shared_memory_row_size + d_k] * Q[((n * H + h) * S + s+1) * d_K + d_k];
        }

        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        // if (d_k == 0) {
        //     T sum_ = 0;
        //     for (int i = 0; i < d_K; i++) {
        //         sum_ += shared_memory[d_K + i];
        //     }
        //     output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        // }

        T out1 = blockReduce<T>(shared_memory[d_K + d_k]);
        T out2 = blockReduce<T>(shared_memory[shared_memory_row_size + d_K + d_k]);
        if (d_k == 0) {
            output[((n * H + h) * S + s) * d_V + d_v] = out1;
        }
        else if (d_k == 1) {
            output[((n * H + h) * S + s+1) * d_V + d_v] = out2;
        }
    }
}



// Wrapper function to orchestrate the computation
template<typename T>
void forward_call_double_over_s(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;

    // What is the maximum number of threads per block?
    int max_threads_per_block = 1024;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_threads_per_block, forward_kernel<T>, block_size, 0);

    // Get the number of blocks we can do in parallel - the number of times the dimension divides the number of threads
    int num_blocks = floor(max_threads_per_block / d_V);
    num_blocks = 1;

    // dim3 grid((int)(d_V/num_blocks), N, H); // Note that d_V is first as it is the fastest changing dimension
    // dim3 block(d_K, num_blocks); // Note that d_K is first as it is the fastest changing dimension
    // dim3 block(max_threads_per_block);
    dim3 grid(d_V, N, H);
    dim3 block(d_K);
    forward_kernel_double_over_s<T><<<grid, block, 2*d_K*sizeof(T) * 2, stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size);
}
















































template <typename T>
__inline__ __device__ T warpReduceSum_dimblock(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync_(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduce_dimblock(T val) {
    static __shared__ T shared[32]; // Assuming a maximum of 32 warps per block
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Reduce within each warp
    val = warpReduceSum_dimblock(val);

    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = val;

    __syncthreads();

    // Ensure we only proceed with the first warp for final reduction
    // if (threadIdx.x < blockDim.x / warpSize) val = shared[lane];else val = 0;
    // The ? operator does not have divergence
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : (T)0;
    if (wid == 0) val = warpReduceSum_dimblock(val); // Final reduce within the first warp

    __syncthreads();

    return val;
}


template<typename T>
__global__ void forward_kernel_dimblock(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    const int block_size) {
    
    int n = blockIdx.y; // Batch index
    int h = blockIdx.z; // Head index
    int d_v = blockIdx.x; // Dimension index within d_V

    int d_k = threadIdx.x; // Dimension index within d_k
    int d_v_offset = threadIdx.y; // Dimension index within the block corresponding to d_v
    int blk_size = blockDim.y; // Block size


    // // Ensure we are within bounds
    // if (d_k >= d_K || d_v >= d_V) {
    //     return;
    // }


    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);


    // Size of shared memory row
    int shared_memory_row_size = 2 * d_K;
    int shared_memory_offset = d_v_offset * shared_memory_row_size;


    // Initialize the shared memory to 0
    shared_memory[shared_memory_offset + d_k] = 0;



    // Iterate over the entire sequence
    for (int s = 0; s < S; s++) {
        // Wait for all threads to finish the previous iteration
        // __syncthreads();

        if (d_k < d_K) {
            // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_v, d_k]
            shared_memory[shared_memory_offset + d_k] += V[((n * H + h) * S + s) * d_V + d_v * blk_size + d_v_offset] * K[((n * H + h) * S + s) * d_K + d_k];
            __syncthreads();

            // 2: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
            shared_memory[shared_memory_offset + d_K + d_k] = shared_memory[shared_memory_offset + d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
        }

        // 3: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        if (d_k == 0) {
            T sum_ = 0;
            for (int i = 0; i < d_K; i++) {
                sum_ += shared_memory[shared_memory_offset + d_K + i];
            }
            output[((n * H + h) * S + s) * d_V + d_v * blk_size + d_v_offset] = sum_;
        }

        // T out = blockReduce_dimblock<T>(shared_memory[(num_blocks + d_v_offset) * d_K + d_k]);
        // if (d_k == 0) {
        //     output[((n * H + h) * S + s) * d_V + d_v] = out;
        // }
    }
}



// Wrapper function to orchestrate the computation
template<typename T>
void forward_call_dimblock(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;

    // What is the maximum number of threads per block?
    int max_threads_per_block = 1024;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_threads_per_block, forward_kernel<T>, block_size, 0);

    // Get the number of blocks we can do in parallel - the number of times the dimension divides the number of threads
    int num_blocks = floor(max_threads_per_block / d_V);
    num_blocks = 4;

    // dim3 grid((int)(d_V/num_blocks), N, H); // Note that d_V is first as it is the fastest changing dimension
    // dim3 block(d_K, num_blocks); // Note that d_K is first as it is the fastest changing dimension
    // dim3 block(max_threads_per_block);
    dim3 grid(floor((float)d_V/(float)num_blocks), N, H);
    dim3 block(d_K, num_blocks);
    forward_kernel_dimblock<T><<<grid, block, 2*d_K*sizeof(T)*num_blocks, stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size);
}
































template <typename T>
__inline__ __device__ T warpReduceSum_seqblock(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync_(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduce_seqblock(T val) {
    static __shared__ T shared[32]; // Assuming a maximum of 32 warps per block
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Reduce within each warp
    val = warpReduceSum_seqblock(val);

    // Write reduced value to shared memory
    if (lane == 0) shared[wid] = val;

    __syncthreads();

    // Ensure we only proceed with the first warp for final reduction
    // if (threadIdx.x < blockDim.x / warpSize) val = shared[lane];else val = 0;
    // The ? operator does not have divergence
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : (T)0;
    if (wid == 0) val = warpReduceSum_seqblock(val); // Final reduce within the first warp

    __syncthreads();

    return val;
}


template<typename T>
__global__ void forward_kernel_seqblock(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    const int block_size) {
    
    int n = blockIdx.y; // Batch index
    int h = blockIdx.z; // Head index
    int d_v = blockIdx.x; // Dimension index within d_V

    int d_k = threadIdx.x; // Dimension index within d_k
    int s_offset = threadIdx.y; // Dimension index within the block corresponding to s
    int blk_size = blockDim.y; // Block size


    // // Ensure we are within bounds
    // if (d_k >= d_K || d_v >= d_V) {
    //     return;
    // }


    // Allocate shared memory - d_K total elements
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);


    // Size of shared memory row
    int shared_memory_row_size = 2 * d_K;
    int shared_memory_offset = s_offset * shared_memory_row_size;


    // Initialize the shared memory to 0
    shared_memory[shared_memory_offset + d_k] = 0;



    // Iterate over the entire sequence
    for (int s = s_offset; s < S; s+=blk_size) {
        // Wait for all threads to finish the previous iteration
        // __syncthreads();

        if (d_k < d_K) {
            // 1: Each thread computes V[:, :, s, d_v] * K[:, :, s, d_k] and adds it to shared[d_v, d_k]
            shared_memory[shared_memory_offset + d_k] += V[((n * H + h) * S + s) * d_V + d_v] * K[((n * H + h) * S + s) * d_K + d_k];
            __syncthreads();

            // 2: Once all VK have been calculated, do a cumulative sum according to the sequence
            for (int s_ = 0; s_ < s_offset; s_++) {
                shared_memory[shared_memory_offset + d_k] += shared_memory[s_ * shared_memory_row_size + d_k];
            }

            // 3: Multiply shared[d_k] by Q[:, :, s, d_k] and store it in the second half of the shared memory shared[d_K + d_k]
            shared_memory[shared_memory_offset + d_K + d_k] = shared_memory[shared_memory_offset + d_k] * Q[((n * H + h) * S + s) * d_K + d_k];
        }

        // 4: Sum all the elements in shared[d_K + d_k] and store it in output[:, :, s, d_v]
        __syncthreads();
        if (d_k == 0) {
            T sum_ = 0;
            for (int i = 0; i < d_K; i++) {
                sum_ += shared_memory[shared_memory_offset + d_K + i];
            }
            output[((n * H + h) * S + s) * d_V + d_v] = sum_;
        }

        // 5: Update the shared memory such that all s_offsets have the saem value
        //    as the last s_offset
        __syncthreads();
        shared_memory[shared_memory_offset + d_k] = shared_memory[blk_size * shared_memory_row_size + d_k];

        // T out = blockReduce_seqblock<T>(shared_memory[(num_blocks + d_v_offset) * d_K + d_k]);
        // if (d_k == 0) {
        //     output[((n * H + h) * S + s) * d_V + d_v] = out;
        // }
    }
}



// Wrapper function to orchestrate the computation
template<typename T>
void forward_call_seqblock(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;

    // What is the maximum number of threads per block?
    int max_threads_per_block = 1024;
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_threads_per_block, forward_kernel<T>, block_size, 0);

    // Get the number of blocks we can do in parallel - the number of times the dimension divides the number of threads
    int num_blocks = floor(max_threads_per_block / d_V);
    num_blocks = 4;

    // dim3 grid((int)(d_V/num_blocks), N, H); // Note that d_V is first as it is the fastest changing dimension
    // dim3 block(d_K, num_blocks); // Note that d_K is first as it is the fastest changing dimension
    // dim3 block(max_threads_per_block);
    dim3 grid(d_V, N, H);
    dim3 block(d_K, num_blocks);
    forward_kernel_dimblock<T><<<grid, block, 2*d_K*sizeof(T)*num_blocks, stream>>>(Q, K, V, output, N, H, S, d_V, d_K, block_size);
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
        forward_call_double_over_d__v_cache<scalar_t>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, H, S, D, block_size);
    }));

    // Device syc for debugging
    cudaDeviceSynchronize();

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