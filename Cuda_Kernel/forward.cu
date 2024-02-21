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




// For debugging
#include <fstream>
template<typename T>
void writeTensorToFile(const std::string& filename, const T* tensorData, const std::vector<int>& shape) {
    std::ofstream file(filename, std::ios::binary | std::ios::out);

    // Write the shape
    int dimensions = shape.size();
    file.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    for (int dim : shape) {
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    }

    // Get the number of elements in the tensor
    int numElements = 1;
    for (int dim : shape) numElements *= dim;

    // Allocate host memory to copy the tensor data
    size_t numBytes = numElements * sizeof(T);
    T* hostData = new T[numElements];

    // Copy the tensor data to host
    cudaMemcpy(hostData, tensorData, numBytes, cudaMemcpyDeviceToHost);

    // Write the tensor data to file
    file.write(reinterpret_cast<const char*>(hostData), sizeof(T) * numElements);

    // Close the file and free the host memory
    file.close();
    free(hostData);
}







// Used to do (V.unsqueeze(-1)*K.unsqueeze(-2)) for each position in the sequence
template<typename T>
__global__ void compute_outer_products(
    const T* K, const T* V, T* VK,
    int N, int H, int S, int d_V, int d_K, int s, int block_size, int BS) {
    int n = floor((float)blockIdx.x / (float)H); // Batch index
    int h = blockIdx.x % H; // Head index
    int blk_idx = blockIdx.y; // Dimension index within the sequence
    int d_v = blockIdx.z; // Dimension index within d_V
    int d_k = threadIdx.x; // Dimension index within d_k

    // Ensure we are within bounds for the d_V dimension and d_K dimension
    if (d_v < d_V && d_k < d_K) {
        // Compute indices for V and K at the current block in the sequence
        int indexV = ((n * H + h) * S + s + blk_idx) * d_V + d_v;
        int indexK = ((n * H + h) * S + s + blk_idx) * d_K + d_k;

        // Do the outer product between V and K
        T product = V[indexV] * K[indexK];

        // Iterate over all blocks with a block index greater than this one
        // and add the product to the VK tensor, thus doing a cumulative sum
        // in the shared memory.
        for (int i = blk_idx; i < BS; i++) {
            int indexVK = ((((n * H + h) * block_size + i) * d_V) + d_v) * d_K + d_k;

            // Add the product to the VK tensor
            AtomicAdd_(&VK[indexVK], product);
        }
    }
}




template<typename T>
__global__ void matrix_multiply_kernel(
    const T* Q, T* VK, T* output,
    int N, int H, int S, int d_V, int d_K, int s, int block_size, int BS) {
    int n = floor((float)blockIdx.x / (float)H); // Batch index
    int h = blockIdx.x % H; // Head index
    int blk_idx = blockIdx.y; // Dimension index within the sequence
    int d_v = blockIdx.z; // Dimension index within d_V
    int d_k = threadIdx.x; // Dimension index within d_k


    // Allocate shared memory for the cumulative sum
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);


    // Ensure we are within bounds for the d_V dimension and d_K dimension
    if (d_v < d_V && d_k < d_K) {
        // Compute indices for Q and VK. Note that VK does not vary with s,
        // so we use a fixed sequence index (effectively 0) for VK.
        int indexQ = ((n * H + h) * S + s + blk_idx) * d_K + d_k;

        // For VK, since it's (N, H, 1, d_V, d_K), we don't include 's' in its index calculation
        int indexVK = (((n * H + h) * block_size + blk_idx) * d_V + d_v) * d_K + d_k;


        // Multiply the Q and VK tensors and accumulate the sum in shared memory
        shared_memory[d_k] = Q[indexQ] * VK[indexVK];

        // Wait for all threads to finish writing to shared memory
        __syncthreads();

        // Only one thread sums all the elements in shared memory and stores the result in output
        if (d_k == 0) {
            int indexOutput = ((n * H + h) * S + s + blk_idx) * d_V + d_v;

            T sum_ = 0;
            for (int i = 0; i < d_K; i++) {
                sum_ += shared_memory[i];
            }
            
            AtomicAdd_(&output[indexOutput], sum_);
        }

        // Since each position in VK is only access once, we can copy
        // the contents of the last block to this one.
        // This ensures the cumulative sum is correct for the next block.
        // Only do this copy if the current block is not the last one
        if (blk_idx < BS-1) {
            VK[indexVK] = VK[(((n * H + h) * block_size + BS-1) * d_V + d_v) * d_K + d_k];
        }
    }
}




template<typename T>
void compute_attention(
    const T* Q, const T* K, const T* V,
    T* output,
    T* VK,
    int N, int H, int S, int d_V, int d_K,
    const int block_size,
    cudaStream_t stream = 0) {

    // writeTensorToFile("Q.bin", Q, {N, H, S, d_K});
    // writeTensorToFile("K.bin", K, {N, H, S, d_K});
    // writeTensorToFile("V.bin", V, {N, H, S, d_V});

    // Iterate over the sequence dimension and compute the outer product
    for (int s = 0; s < S; s+=block_size) {
        // Block size cannot exceed the sequence length
        int BS = min(block_size, S-s);

        // Compute the cumulative product between V and K
        // up to block_size positions in the sequence
        //   Grid over N, H, and value dimension. Assuming the block size is small
        //      we can use this as the x index in the thread and y as the d_K index
        //   Threads over the number of blocks and the d_K dimension
        //   No shared memory
        //   Stream is the CUDA stream where the kernel will be executed
        dim3 grid(N*H, BS, d_V);
        compute_outer_products<T><<<grid, d_K, 0, stream>>>(K, V, VK, N, H, S, d_V, d_K, s, block_size, BS);

        // // Wait for the kernel to complete
        // cudaDeviceSynchronize();

        // Product between Q at position s and VK
        //   Grid over N, H, and value dimension. Assuming the block size is small
        //      we can use this as the x index in the thread and y as the d_K index
        //   Threads over the number of blocks and the d_K dimension
        //   Shared memory is used to accumulate the sum
        //   stream - This is the CUDA stream where the kernel will be executed
        matrix_multiply_kernel<T><<<grid, d_K, d_K*sizeof(T), stream>>>(Q, VK, output, N, H, S, d_V, d_K, s, block_size, BS);

        // // Wait for the kernel to complete
        // cudaDeviceSynchronize();
    }

    // writeTensorToFile("VK.bin", VK, {N, H, 1, d_V, d_K});
    // writeTensorToFile("output.bin", output, {N, H, S, d_V});
}



// Wrapper function to orchestrate the computation
template<typename T>
void forward_call(
    const T* Q, const T* K, const T* V, T* output, T* VK,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {
    compute_attention<T>(Q, K, V, output, VK, N, H, S, D, D, block_size, stream);
}





// // CUDA forward declarations
// void compute_and_contract_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);
// void compute_and_contract(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);

// C++ interface
template<typename dtype_>
torch::Tensor forward_(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V, const int8_t block_size) {
    // Check tensor requirements, e.g., dtype, device, etc.
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");

    // Get tensor dimensions
    int N = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    // Ouput tensor
    auto output = torch::zeros({N, H, S, D}, torch::TensorOptions().dtype(Q.scalar_type()).device(Q.device()));
    // auto output = K_orig;

    // Allocate memory for the intermediate tensors
    auto VK = torch::zeros({N, H, block_size, D, D}, torch::TensorOptions().dtype(Q.scalar_type()).device(Q.device()));

    // writeTensorToFile("Q.bin", Q.data_ptr<float>(), {N, H, S, D});
    // writeTensorToFile("K.bin", K_orig.data_ptr<float>(), {N, H, S, D});
    // writeTensorToFile("V.bin", V_orig.data_ptr<float>(), {N, H, S, D});

    // // Unsqueeze K along the last dimension and V along the second-to-last dimension
    // auto K = K_orig.unsqueeze(-1); // (N, H, S, D, 1)
    // auto V = V_orig.unsqueeze(-2); // (N, H, S, 1, D)
    // Unsqueeze not needed as I am making the kernel hehe UwU

    // Ensure the tensors are contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    // c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan.cpp
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)Q.get_device()};
    // c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Call the CUDA kernel
    forward_call<dtype_>(
        Q.data_ptr<dtype_>(),
        K.data_ptr<dtype_>(),
        V.data_ptr<dtype_>(),
        output.data_ptr<dtype_>(),
        VK.data_ptr<dtype_>(),
        N, H, S, D, block_size);
    // forward_call<dtype_>(
    //     at::autocast::cached_cast(at::kHalf, Q).data_ptr<dtype_>(),
    //     at::autocast::cached_cast(at::kHalf, K).data_ptr<dtype_>(),
    //     at::autocast::cached_cast(at::kHalf, V).data_ptr<dtype_>(),
    //     at::autocast::cached_cast(at::kHalf, output).data_ptr<dtype_>(),
    //     at::autocast::cached_cast(at::kHalf, VK).data_ptr<dtype_>(),
    //     N, H, S, D, block_size);

    // writeTensorToFile("output.bin", output.data_ptr<float>(), {N, H, S, D});

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
