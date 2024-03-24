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





template<typename T, int warp_size>
__global__ void backward_kernel(
    const T* Q, const T* K, const T* V, const T* G,
    T* K_grad, T* V_grad,
    int N, int H, int S, const int D) {
    
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d_outer = blockIdx.z; // Index within the douter D dimension
    int d_inner_block = threadIdx.x; // Thread index within the inner D dimension

    // Block size for each thread in the warp
    int block_size = (int)D/warp_size;
    // Starting index of the inner D dimension
    int d_inner_start = d_inner_block * block_size;
    // Ending index of the inner D dimension
    int d_inner_end = min(d_inner_start + block_size, D);



    // Allocate shared memory. This will store values for prev_grad[*, d_inner] * Q[*, d_outer] for all d_inner
    // and prev_grad[*, d_outer] * Q[*, d_inner] for all d_inner
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];T *CumsumArr_G = reinterpret_cast<T *>(shared_memory_uchar);
    memset(CumsumArr_G, 0, 2*D*sizeof(T));
    T* CumsumArr_Q = &CumsumArr_G[D];

    // Iterate over the entire sequence
    for (int s = S-1; s >= 0; s--) {
        // Initialize the sum over the inner dimension over G and Q to be zero
        T sum_over_G = 0;
        T sum_over_Q = 0;

        // Load in the value of G and Q of the outer dimension for this part of the sequence
        T G_cache = G[((n * H + h) * S + s) * D + d_outer];
        T Q_cache = Q[((n * H + h) * S + s) * D + d_outer];

        // Iterate over the entire block of the dimension that this thread is working on
        #pragma unroll
        for (int d_inner = d_inner_start; d_inner < d_inner_end; d_inner+=4) {
            // Compute the sum and store it in the cumsum array
            // G[s, d_outer] * Q[s, d_inner] and G[s, d_inner] * Q[s, d_outer]
            CumsumArr_G[d_inner] += G[((n * H + h) * S + s) * D + d_inner] * Q_cache;
            CumsumArr_G[d_inner+1] += G[((n * H + h) * S + s) * D + d_inner+1] * Q_cache;
            CumsumArr_G[d_inner+2] += G[((n * H + h) * S + s) * D + d_inner+2] * Q_cache;
            CumsumArr_G[d_inner+3] += G[((n * H + h) * S + s) * D + d_inner+3] * Q_cache;
            CumsumArr_Q[d_inner] += G_cache * Q[((n * H + h) * S + s) * D + d_inner];
            CumsumArr_Q[d_inner+1] += G_cache * Q[((n * H + h) * S + s) * D + d_inner+1];
            CumsumArr_Q[d_inner+2] += G_cache * Q[((n * H + h) * S + s) * D + d_inner+2];
            CumsumArr_Q[d_inner+3] += G_cache * Q[((n * H + h) * S + s) * D + d_inner+3];

            // Multiply the cumsum value with the value or keys and add it to the sum
            // V[s, d_inner] * (G[s, d_inner] * Q[s, d_outer]) and K[s, d_inner] * (G[s, d_outer] * Q[s, d_inner])
            sum_over_G += CumsumArr_G[d_inner] * V[((n * H + h) * S + s) * D + d_inner]
                    + CumsumArr_G[d_inner+1] * V[((n * H + h) * S + s) * D + d_inner+1]
                    + CumsumArr_G[d_inner+2] * V[((n * H + h) * S + s) * D + d_inner+2]
                    + CumsumArr_G[d_inner+3] * V[((n * H + h) * S + s) * D + d_inner+3];
            sum_over_Q += CumsumArr_Q[d_inner] * K[((n * H + h) * S + s) * D + d_inner]
                    + CumsumArr_Q[d_inner+1] * K[((n * H + h) * S + s) * D + d_inner+1]
                    + CumsumArr_Q[d_inner+2] * K[((n * H + h) * S + s) * D + d_inner+2]
                    + CumsumArr_Q[d_inner+2] * K[((n * H + h) * S + s) * D + d_inner+3];
        }

        __syncthreads();

        // Reduce the sum across the block. Easy since we only have 32 threads in the block which is a single warp
        #pragma unroll
        for (int offset = (int)warp_size/2; offset > 0; offset /= 2) {
            sum_over_G += __shfl_down_sync_(0xFFFFFFFF, sum_over_G, offset);
            sum_over_Q += __shfl_down_sync_(0xFFFFFFFF, sum_over_Q, offset);
        }

        __syncthreads();

        // Store the sum in the output
        if (threadIdx.x == 0) {
            K_grad[((n * H + h) * S + s) * D + d_outer] = sum_over_G;
            V_grad[((n * H + h) * S + s) * D + d_outer] = sum_over_Q;
        }
        __syncthreads();
    }

}

// Wrapper function to orchestrate the computation
template<typename T>
void backward_call(
    const T* Q, const T* K, const T* V, const T* previous_grad,
    T* K_grad, T* V_grad, 
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {

    const int warp_size = 32;
    dim3 grid(N, H, D);
    dim3 block(warp_size);
    // Shared memory is the cumsum over G and Q
    backward_kernel<T, warp_size><<<grid, block, 2*D*sizeof(T), stream>>>(Q, K, V, previous_grad, K_grad, V_grad, N, H, S, D);
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
    auto K_grad = torch::empty({N, H, S, D}, torch::TensorOptions().dtype(data_type).device(Q.device()));
    auto V_grad = torch::empty({N, H, S, D}, torch::TensorOptions().dtype(data_type).device(Q.device()));

    // https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)Q.get_device()};

    // Using AT_DISPATCH_FLOATING_TYPES_AND_HALF to handle different data types
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(Q.scalar_type(), "backward_cuda", ([&] {
        backward_call<scalar_t>(
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
