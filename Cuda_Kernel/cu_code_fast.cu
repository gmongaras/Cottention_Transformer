#include <cuda_runtime.h> // For cudaMemcpy and cudaFree
#include <torch/torch.h>
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






// Used to do (V.unsqueeze(-1)*K.unsqueeze(-2)) for each position in the sequence
template<typename T>
__global__ void compute_outer_products(
    const T* K, const T* V, T* VK,
    int N, int H, int S, int d_V, int d_K, int s, int block_size, int BS) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d_v = blockIdx.z; // Dimension index within d_V
    int blk_idx = threadIdx.x; // Dimension index within the sequence
    int d_k = threadIdx.y; // Dimension index within d_k

    int nHh = n * H + h;
    int nHhSsb = (nHh * S + s + blk_idx);

    // // Ensure we are within bounds for the d_V dimension and d_K dimension
    // if (d_v < d_V && d_k < d_K) {
    // Do the outer product between V and K
    T product = V[nHhSsb * d_V + d_v] * K[nHhSsb * d_K + d_k];

    // Iterate over all blocks with a block index greater than this one
    // and add the product to the VK tensor, thus doing a cumulative sum
    // in the shared memory.
    for (int i = blk_idx; i < BS; i++) {
        // Add the product to the VK tensor
        AtomicAdd_(&VK[(((nHh * block_size + i) * d_V) + d_v) * d_K + d_k], product);
    }
    // }
}


// Used to do (Q*VK) for each position in the sequence
template<typename T>
__global__ void matrix_multiply_kernel(
    const T* Q, T* VK, T* output,
    int N, int H, int S, int d_V, int d_K, int s, int block_size, int BS) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d_v = blockIdx.z; // Dimension index within d_V
    int blk_idx = threadIdx.x; // Sequence index
    int d_k = threadIdx.y; // Dimension index within d_K

    int nHh = n * H + h;
    int nHhSsb = (nHh * S + s + blk_idx);

    // // Ensure we are within bounds for the d_V dimension and d_K dimension
    // if (d_v < d_V && d_k < d_K) {
    // For VK, since it's (N, H, 1, d_V, d_K), we don't include 's' in its index calculation
    int indexVK = ((nHh * block_size + blk_idx) * d_V + d_v) * d_K + d_k;

    // Element-wise multiplication and add to shared memory
    // Write the accumulated sum to the output tensor
    AtomicAdd_(&output[nHhSsb * d_V + d_v], Q[nHhSsb * d_K + d_k] * VK[indexVK]);

    // Since each position in VK is only access once, we can copy
    // the contents of the last block to this one.
    // This ensures the cumulative sum is correct for the next block.
    VK[indexVK] = VK[((nHh * block_size + BS-1) * d_V + d_v) * d_K + d_k];
    // }
}




template<typename T>
void compute_attention(
    const T* Q, const T* K, const T* V,
    T* output,
    T* VK,
    int N, int H, int S, int d_V, int d_K,
    const int block_size,
    cudaStream_t stream = 0) {
    // Grid for the matrix multiplication kernel
    // One block per batch-dimension index, head-dimension index, and both dimensions of VK
    dim3 grid(N, H, d_V);

    // Intermediate tensor to store the product between V and K
    // for block_size positions in the sequence.
    // The shape is (N, H, block_size, d_V, d_K)
    // T* VK;
    // Initialize to zeros
    // cudaMalloc(&VK, N * H * block_size * d_V * d_K * sizeof(T));
    // cudaMemset(VK, 0, N * H * block_size * d_V * d_K * sizeof(T));
    // Torch tensor VK

    // Iterate over the sequence dimension and compute the outer product
    for (int s = 0; s < S; s+=block_size) {
        // Block size cannot exceed the sequence length
        u_int8_t BS = min(block_size, S-s);

        // Compute the cumulative product between V and K
        // up to block_size positions in the sequence
        //   Grid over N, H, and value dimension. Assuming the block size is small
        //      we can use this as the x index in the thread and y as the d_K index
        //   Threads over the number of blocks and the d_K dimension
        //   No shared memory
        //   Stream is the CUDA stream where the kernel will be executed
        compute_outer_products<T><<<grid, {BS, d_K}, 0, stream>>>(K, V, VK, N, H, S, d_V, d_K, s, block_size, BS);

        // // Wait for the kernel to complete
        // cudaDeviceSynchronize();

        // Product between Q at position s and VK
        //   Grid over N, H, and value dimension. Assuming the block size is small
        //      we can use this as the x index in the thread and y as the d_K index
        //   Threads over the number of blocks and the d_K dimension
        //   0 - This is the shared memory size, which is not used in this kernel
        //   stream - This is the CUDA stream where the kernel will be executed
        matrix_multiply_kernel<T><<<grid, {BS, d_K}, 0, stream>>>(Q, VK, output, N, H, S, d_V, d_K, s, block_size, BS);

        // // Wait for the kernel to complete
        // cudaDeviceSynchronize();
    }

    // Free the intermediate tensor
    // cudaFreeAsync(VK, stream);

    // // Delete the intermediate tensor async
    // cudaStreamSynchronize(stream);
    // cudaFreeAsync(VK);
}



// Wrapper function to orchestrate the computation
template<typename T>
void compute_and_contract(
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
torch::Tensor compute_and_contract_call(const torch::Tensor& Q, const torch::Tensor& K_orig, const torch::Tensor& V_orig, const int block_size) {
    // // Check tensor requirements, e.g., dtype, device, etc.
    // TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    // TORCH_CHECK(K_orig.device().is_cuda(), "K must be a CUDA tensor");
    // TORCH_CHECK(V_orig.device().is_cuda(), "V must be a CUDA tensor");
    // TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    // Get tensor dimensions
    int N = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    // Ouput tensor
    auto output = torch::zeros({N, H, S, D}, Q.options());

    // Allocate memory for the intermediate tensors if VK is nullptr
    auto VK = torch::zeros({N, H, block_size, D, D}, Q.options());

    // Unsqueeze K along the last dimension and V along the second-to-last dimension
    auto K = K_orig.unsqueeze(-1); // (N, H, S, D, 1)
    auto V = V_orig.unsqueeze(-2); // (N, H, S, 1, D)

    // Call the CUDA kernel
    compute_and_contract<dtype_>(
        Q.data_ptr<dtype_>(),
        K.data_ptr<dtype_>(),
        V.data_ptr<dtype_>(),
        output.data_ptr<dtype_>(),
        VK.data_ptr<dtype_>(),
        N, H, S, D, block_size);


    return output;
}


// torch::Tensor compute_and_contract_call(const torch::Tensor& Q, const torch::Tensor& K_orig, const torch::Tensor& V_orig, torch::Tensor& output, torch::Tensor& VK, const int block_size) {
//     // // Check tensor requirements, e.g., dtype, device, etc.
//     // TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
//     // TORCH_CHECK(K_orig.device().is_cuda(), "K must be a CUDA tensor");
//     // TORCH_CHECK(V_orig.device().is_cuda(), "V must be a CUDA tensor");
//     // TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

//     // Get tensor dimensions
//     int N = Q.size(0);
//     int H = Q.size(1);
//     int S = Q.size(2);
//     int D = Q.size(3);

//     // // Ouput tensor
//     // auto output = torch::zeros({N, H, S, D}, Q.options());

//     // // Allocate memory for the intermediate tensors
//     // auto VK = torch::zeros({N, H, block_size, D, D}, Q.options());

//     // Unsqueeze K along the last dimension and V along the second-to-last dimension
//     auto K = K_orig.unsqueeze(-1); // (N, H, S, D, 1)
//     auto V = V_orig.unsqueeze(-2); // (N, H, S, 1, D)


//     // Call the CUDA kernel
//     compute_and_contract<float>(
//         Q.data_ptr<float>(),
//         K.data_ptr<float>(),
//         V.data_ptr<float>(),
//         output.data_ptr<float>(),
//         VK.data_ptr<float>(),
//         N, H, S, D, block_size);

//     // Deallocate memory for the intermediate tensors
//     VK.~Tensor();

//     return output;
// }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float32", &compute_and_contract_call<float>);
    m.def("float16", &compute_and_contract_call<at::Half>);
    m.def("bfloat16", &compute_and_contract_call<at::BFloat16>);
}