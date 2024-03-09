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
__inline__ __device__ void warpReduceSum(T& val1, T& val2) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val1 += __shfl_down_sync_(0xffffffff, val1, offset);
        val2 += __shfl_down_sync_(0xffffffff, val2, offset);
    }
}

template <typename T>
__inline__ __device__ void blockReduce(T& val1, T& val2) {
    static __shared__ T shared1[32]; // For the first value
    static __shared__ T shared2[32]; // For the second value
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    // Reduce within each warp for both values
    warpReduceSum(val1, val2);

    // Write reduced values to shared memory
    if (lane == 0) {
        shared1[wid] = val1;
        shared2[wid] = val2;
    }

    __syncthreads();

    // Use only the first warp for the final reduction
    val1 = (threadIdx.x < blockDim.x / warpSize) ? shared1[lane] : (T)0;
    val2 = (threadIdx.x < blockDim.x / warpSize) ? shared2[lane] : (T)0;
    if (wid == 0) {
        warpReduceSum(val1, val2); // Final reduce within the first warp
    }

    __syncthreads();
}






template<typename T>
__inline__ __device__ void backward_kernel_double_over_d__v_cache_one_call(
    const T* Q, const T* K, const T* V, const T* prev_grad,
    T* K_grad, T* V_grad,
    T* shared_memory_cumsum_GQ, T* shared_memory_reduce_Vgrad,
    T* shared_memory_cumsum_QG, T* shared_memory_reduce_Kgrad,
    T* shared_memory_GCache, T* shared_memory_QCache,
    int s, int S, int n, int N, int h, int H, int d_g, int d_G, int d_q, int d_Q ) {
    

    // Wait for all threads to finish the previous iteration
    __syncthreads();

    // 1: Each thread computes G[:, :, s, d_g] * Q[:, :, s, d_q] and adds it to shared[d_q] (block 1)
    //    and each thread computes Q[:, :, s, d_g] * G[:, :, s, d_v] and adds it to shared[2*d_Q + d_q] (block 3)
    shared_memory_cumsum_GQ[d_q] += prev_grad[((n * H + h) * S + s) * d_G + d_g] * Q[((n * H + h) * S + s) * d_Q + d_q];
    shared_memory_cumsum_QG[d_q] += Q[((n * H + h) * S + s) * d_G + d_g] * prev_grad[((n * H + h) * S + s) * d_Q + d_q];

    // 2: Multiply shared[d_q] by K[:, :, s, d_q] and store it in the second part of the shared memory shared[d_Q + d_q] for the V gradient
    //    and multiply shared[2*d_Q + d_q] by V[:, :, s, d_q] and store it in the third part of the shared memory shared[3*d_Q + d_q] for the K gradient
    shared_memory_reduce_Vgrad[d_q] = shared_memory_cumsum_GQ[d_q] * K[((n * H + h) * S + s) * d_Q + d_q];
    shared_memory_reduce_Kgrad[d_q] = shared_memory_cumsum_QG[d_q] * V[((n * H + h) * S + s) * d_Q + d_q];

    // 3: Thread 1 sums all the elements in block 1 and store it in grad_V[:, :, s, d_g]
    //    Thread 2 sums all the elements in block 4 and store it in grad_K[:, :, s, d_g]
    __syncthreads();
    // if (d_q == 0) {
    //     T sum_ = 0;
    //     for (int i = 0; i < d_Q; i++) {
    //         sum_ += shared_memory_reduce_Vgrad[i];
    //     }
    //     V_grad[((n * H + h) * S + s) * d_G + d_g] = sum_;
    // }
    // else if (d_q == 1) {
    //     T sum_ = 0;
    //     for (int i = 0; i < d_Q; i++) {
    //         sum_ += shared_memory_reduce_Kgrad[i];
    //     }
    //     K_grad[((n * H + h) * S + s) * d_G + d_g] = sum_;
    // }

    // Array of two values to store the output of the blockReduce
    blockReduce(shared_memory_reduce_Vgrad[d_q], shared_memory_reduce_Kgrad[d_q]);
    if (threadIdx.x == 0) {
        V_grad[((n * H + h) * S + s) * d_G + d_g] = shared_memory_reduce_Vgrad[d_q];
        K_grad[((n * H + h) * S + s) * d_G + d_g] = shared_memory_reduce_Kgrad[d_q];
    }
}



template<typename T, unsigned int inner_loop_size>
__inline__ __device__ void backward_kernel_double_over_d__v_cache_loop_inner(
    const T* Q, const T* K, const T* V, const T* prev_grad,
    T* K_grad, T* V_grad,
    T* shared_memory_cumsum_GQ, T* shared_memory_reduce_Vgrad,
    T* shared_memory_cumsum_QG, T* shared_memory_reduce_Kgrad,
    T* shared_memory_GCache, T* shared_memory_QCache,
    int s_start, int S, int n, int N, int h, int H, int d_g, int d_G, int d_q, int d_Q ) {
    #pragma unroll
    for (int i = 0; i < inner_loop_size; i++) {
        backward_kernel_double_over_d__v_cache_one_call<T>(
            Q, K, V, prev_grad, K_grad, V_grad, shared_memory_cumsum_GQ, shared_memory_reduce_Vgrad, shared_memory_cumsum_QG, shared_memory_reduce_Kgrad, shared_memory_GCache, shared_memory_QCache, s_start-i, S, n, N, h, H, d_g, d_G, d_q, d_Q);
    }
    // forward_kernel_double_over_d__v_cache_one_call<T>(
    //         Q, K, V, output, shared_memory_cumsum, shared_memory_reduce, shared_memory_VCache, s_start, S, n, N, h, H, d_k, d_K, d_v, d_V);
    // forward_kernel_double_over_d__v_cache_one_call<T>(
    //         Q, K, V, output, shared_memory_cumsum, shared_memory_reduce, shared_memory_VCache, s_start+1, S, n, N, h, H, d_k, d_K, d_v, d_V);
    // forward_kernel_double_over_d__v_cache_one_call<T>(
    //         Q, K, V, output, shared_memory_cumsum, shared_memory_reduce, shared_memory_VCache, s_start+2, S, n, N, h, H, d_k, d_K, d_v, d_V);
    // forward_kernel_double_over_d__v_cache_one_call<T>(
    //         Q, K, V, output, shared_memory_cumsum, shared_memory_reduce, shared_memory_VCache, s_start+3, S, n, N, h, H, d_k, d_K, d_v, d_V);
    // forward_kernel_double_over_d__v_cache_one_call<T>(
    //         Q, K, V, output, shared_memory_cumsum, shared_memory_reduce, shared_memory_VCache, s_start+4, S, n, N, h, H, d_k, d_K, d_v, d_V);
    // forward_kernel_double_over_d__v_cache_one_call<T>(
    //         Q, K, V, output, shared_memory_cumsum, shared_memory_reduce, shared_memory_VCache, s_start+5, S, n, N, h, H, d_k, d_K, d_v, d_V);
    // forward_kernel_double_over_d__v_cache_one_call<T>(
    //         Q, K, V, output, shared_memory_cumsum, shared_memory_reduce, shared_memory_VCache, s_start+6, S, n, N, h, H, d_k, d_K, d_v, d_V);
    // forward_kernel_double_over_d__v_cache_one_call<T>(
    //         Q, K, V, output, shared_memory_cumsum, shared_memory_reduce, shared_memory_VCache, s_start+7, S, n, N, h, H, d_k, d_K, d_v, d_V);
}





template<typename T, unsigned int inner_loop_size>
__global__ void backward_kernel_double_over_d__v_cache(
    const T* Q, const T* K, const T* V, const T* prev_grad,
    T* K_grad, T* V_grad,
    int N, int H, int S, int d_G, int d_Q,
    const int block_size) {
    
    int n = blockIdx.y; // Batch index
    int h = blockIdx.z; // Head index
    int d_g = blockIdx.x; // Dimension index within d_G
    int d_q = threadIdx.x; // Dimension index within d_Q

    int shared_memory_row_size = 2 * d_Q;


    // // Ensure we are within bounds
    // if (d_k >= d_K || d_v >= d_V) {
    //     return;
    // }


    // Allocate shared memory
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    // extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    // T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];T *shared_memory_cumsum_GQ = reinterpret_cast<T *>(shared_memory_uchar);
    T* shared_memory_reduce_Vgrad = &shared_memory_cumsum_GQ[d_Q];
    T* shared_memory_cumsum_QG = &shared_memory_cumsum_GQ[2*d_Q];
    T* shared_memory_reduce_Kgrad = &shared_memory_cumsum_GQ[3*d_Q];
    T* shared_memory_GCache = &shared_memory_cumsum_GQ[4*d_Q];
    T* shared_memory_QCache = &shared_memory_cumsum_GQ[4*d_Q + S];

    // Initialize the shared memory to 0
    if (d_q < d_Q) {
        shared_memory_cumsum_GQ[d_q] = shared_memory_cumsum_QG[d_q] =  0;
    }



    // Cache the previous gradient and Q values
    for (int s = d_q; s < S; s += d_Q) {
        shared_memory_GCache[s] = prev_grad[((n * H + h) * S + s) * d_G + d_g];
        shared_memory_QCache[s] = Q[((n * H + h) * S + s) * d_Q + d_q];
    }
    __syncthreads();


    // Iterate over the first uneven part of the sequence
    for (int s = S-1; s >= floor((float)S/(float)inner_loop_size)*inner_loop_size; s--) {
        backward_kernel_double_over_d__v_cache_one_call<T>(
            Q, K, V, prev_grad, K_grad, V_grad, shared_memory_cumsum_GQ, shared_memory_reduce_Vgrad, shared_memory_cumsum_QG, shared_memory_reduce_Kgrad, shared_memory_GCache, shared_memory_QCache, s, S, n, N, h, H, d_g, d_G, d_q, d_Q);
    }

    // Iterate over the entire sequence
    for (int s = floor((float)S/(float)inner_loop_size); s > 0; s--) {
        backward_kernel_double_over_d__v_cache_loop_inner<T, inner_loop_size>(
            Q, K, V, prev_grad, K_grad, V_grad, shared_memory_cumsum_GQ, shared_memory_reduce_Vgrad, shared_memory_cumsum_QG, shared_memory_reduce_Kgrad, shared_memory_GCache, shared_memory_QCache, s*inner_loop_size-1, S, n, N, h, H, d_g, d_G, d_q, d_Q);
    }
}



// Wrapper function to orchestrate the computation
template<typename T>
void backward_call_double_over_d__v_cache(
    const T* Q, const T* K, const T* V, const T* prev_grad,
    T* K_grad, T* V_grad,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    int d_V = D;
    int d_K = D;

    // Inner loop size is 8
    const int inner_loop_size = 8;

    // Shared memory has 4 parts:
    //  1: Cumulative sum of G_j * Q over all Q
    //  2: block 1 multiplied by K for the V gradient
    //  3: Cumulative sum of Q_j * G over all G
    //  4: block 3 multiplied by V for the K gradient
    //  5: Cache for the prev_grad
    //  6: Cache for the Q values
    dim3 grid(d_V, N, H);
    dim3 block((int)d_K);
    backward_kernel_double_over_d__v_cache<T, inner_loop_size><<<grid, block, 4*d_K*sizeof(T) + 2*S*sizeof(T), stream>>>(Q, K, V, prev_grad, K_grad, V_grad, N, H, S, D, D, block_size);
}


















































// C++ interface
template<typename dtype_>
std::vector<torch::Tensor> backward_(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V, torch::Tensor& prev_grad, const int8_t block_size) {
    // Check tensor requirements, e.g., dtype, device, etc.
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(prev_grad.device().is_cuda(), "prev_grad must be a CUDA tensor");

    // Get tensor dimensions
    int N = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    // Get the data type, could be auto casted
    auto data_type = at::autocast::is_enabled() && Q.scalar_type() == at::kFloat ? at::kHalf : Q.scalar_type();

    // // Unsqueeze prev_grad along the last dimension and Q along the second-to-last dimension
    // auto prev_grad = prev_grad_orig.unsqueeze(-1); // (N, H, S, D, 1)
    // auto V = V_orig.unsqueeze(-2); // (N, H, S, 1, D)
    // Unsqueeze not needed as I am making the kernel hehe UwU

    // Ensure the tensors are contiguous
    Q = Q.contiguous().to(data_type);
    K = K.contiguous().to(data_type);
    V = V.contiguous().to(data_type);
    prev_grad = prev_grad.contiguous().to(data_type);

    // Ouput tensors, gradient of K and V
    auto K_grad = torch::zeros({N, H, S, D}, torch::TensorOptions().dtype(data_type).device(Q.device()));
    auto V_grad = torch::zeros({N, H, S, D}, torch::TensorOptions().dtype(data_type).device(Q.device()));

    // https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)Q.get_device()};

    // Using AT_DISPATCH_FLOATING_TYPES_AND_HALF to handle different data types
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "backward_cuda", ([&] {
        backward_call_double_over_d__v_cache<scalar_t>(
        Q.data_ptr<scalar_t>(),
        K.data_ptr<scalar_t>(),
        V.data_ptr<scalar_t>(),
        prev_grad.data_ptr<scalar_t>(),
        K_grad.data_ptr<scalar_t>(),
        V_grad.data_ptr<scalar_t>(),
        N, H, S, D, block_size);
    }));

    return {K_grad, V_grad};
}




TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, Autocast, m) {
    m.impl("float32", backward_<float>);
    m.impl("float64", backward_<double>);
    m.impl("float16", backward_<at::Half>);
    try {
        m.impl("bfloat16", backward_<at::BFloat16>);
    } catch (const std::exception& e) {
        std::cout << "GPU does not support bfloat16. Skipping..." << std::endl;
        // std::cerr << "Error: " << e.what() << std::endl;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float32", &backward_<float>);
    m.def("float64", &backward_<double>);
    m.def("float16", &backward_<at::Half>);
    try {
        m.def("bfloat16", &backward_<at::BFloat16>);
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