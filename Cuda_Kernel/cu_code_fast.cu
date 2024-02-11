#include <cuda_runtime.h> // For cudaMemcpy and cudaFree
#include <torch/torch.h>
// #include <torch/extension.h>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <chrono>












// Used to do (V.unsqueeze(-1)*K.unsqueeze(-2))
template<typename T>
__global__ void compute_outer_product_kernel(
    const T* K, const T* V, T* VK,
    int N, int H, int S, int d_V, int d_K, int s) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d_v = blockIdx.z; // Dimension index within d_V (output dimension)
    int d_k = threadIdx.x; // Dimension index within d_V (output dimension)

    int nHh = n * H + h;
    int nHhSs = nHh * S + s;

    // Perform the outer product and add it to VK
    atomicAdd(&VK[(((nHh) * d_V) + d_v) * d_K + d_k], V[nHhSs * d_V + d_v] * K[nHhSs * d_K + d_k]);
}




template<typename T>
__global__ void matrix_multiply_kernel(
    const T* Q, const T* VK, T* output,
    int N, int H, int S, int d_V, int d_K, int s) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d_v = blockIdx.z; // Dimension index within d_V (output dimension)
    int d_k = threadIdx.x; // Dimension index within d_V (output dimension)

    int nHh = n * H + h;
    int nHhSs = nHh * S + s;

    atomicAdd(&output[nHhSs * d_V + d_v], Q[nHhSs * d_K + d_k] * VK[(nHh * d_V + d_v) * d_K + d_k]);
}




template<typename T>
void compute_attention(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    cudaStream_t stream = 0) {
    // Grid for the outer product kernel
    // One block per batch-dimension index and head-dimension index
    dim3 outer_grid(N, H, d_V);

    // Grid for the matrix multiplication kernel
    // One block per batch-dimension index, head-dimension index, and both dimensions of VK
    dim3 grid(N, H, d_V);

    // Intermediate tensor to store the product between V and K
    // at each position in the sequence
    // The shape is (N, H, 1, d_V, d_K)
    T* VK;
    // Initialize to zeros
    cudaMalloc(&VK, N * H * d_V * d_K * sizeof(T));
    cudaMemset(VK, 0, N * H * d_V * d_K * sizeof(T));

    // writeTensorToFile("Q.bin", Q, {N, H, S, d_K});
    // writeTensorToFile("K.bin", K, {N, H, S, d_K});
    // writeTensorToFile("V.bin", V, {N, H, S, d_V});

    // Iterate over the sequence dimension and compute the outer product
    for (int s = 0; s < S; ++s) {
        // Launch the kernel
        // This will compute the outer product between V and K at each position in the sequence
        // and add the result to VK
        compute_outer_product_kernel<T><<<outer_grid, d_K, 0, stream>>>(K, V, VK, N, H, S, d_V, d_K, s);

        // // Wait for the kernel to complete
        // cudaDeviceSynchronize();

        // Product between Q at position s and VK
        // This is the output for the s-th position in the sequence
        // we want d_K threads per block
        matrix_multiply_kernel<T><<<grid, d_K, 0, stream>>>(Q, VK, output, N, H, S, d_V, d_K, s);
        // matrix_multiply_kernel<T><<<grid, d_V, 0, stream>>>(Q, VK, output, N, H, S, d_V, d_K, s);

        // // Wait for the kernel to complete
        // cudaDeviceSynchronize();
    }

    // writeTensorToFile("VK.bin", VK, {N, H, 1, d_V, d_K});
    // writeTensorToFile("output.bin", output, {N, H, S, d_V});

    // Free the intermediate tensor
    cudaFree(VK);
}



// Wrapper function to orchestrate the computation
template<typename T>
void compute_and_contract(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    cudaStream_t stream = 0) {
    compute_attention<T>(Q, K, V, output, N, H, S, D, D, stream);
}





// // CUDA forward declarations
// void compute_and_contract_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);
// void compute_and_contract(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);

// C++ interface
void compute_and_contract_call(const torch::Tensor& Q, const torch::Tensor& K_orig, const torch::Tensor& V_orig, torch::Tensor& output) {
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

    // writeTensorToFile("Q.bin", Q.data_ptr<float>(), {N, H, S, D});
    // writeTensorToFile("K.bin", K_orig.data_ptr<float>(), {N, H, S, D});
    // writeTensorToFile("V.bin", V_orig.data_ptr<float>(), {N, H, S, D});

    // Unsqueeze K along the last dimension and V along the second-to-last dimension
    auto K = K_orig.unsqueeze(-1); // (N, H, S, D, 1)
    auto V = V_orig.unsqueeze(-2); // (N, H, S, 1, D)

    // Call the CUDA kernel
    compute_and_contract<float>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        N, H, S, D);

    // writeTensorToFile("output.bin", output.data_ptr<float>(), {N, H, S, D});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_and_contract", &compute_and_contract_call, "Compute and contract operation");
}