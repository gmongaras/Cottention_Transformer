#include <cuda_runtime.h> // For cudaMemcpy and cudaFree
#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <chrono>






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






// Used to do (V.unsqueeze(-1)*K.unsqueeze(-2))
template<typename T>
__global__ void compute_outer_product_kernel(
    const T* K, const T* V, T* VK,
    int N, int H, int S, int d_V, int d_K, int s) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index

    // Each thread handles a portion of the elements in VK
    int threadId = threadIdx.x;
    int totalThreads = blockDim.x;

    for (int idx = threadId; idx < d_V * d_K; idx += totalThreads) {
        int d_v = idx % d_V; // Correct calculation for Dimension index for V
        int d_k = idx / d_V; // Correct calculation for Dimension index for K

        // Compute indices for V, K, and VK
        int indexV = ((n * H + h) * S + s) * d_V + d_v;
        int indexK = ((n * H + h) * S + s) * d_K + d_k;
        int indexVK = ((((n * H + h) * 1 + 0) * d_V) + d_v) * d_K + d_k; // Use s=0 for VK since it's the accumulation target

        // Perform the outer product and store in VK directly without addition
        VK[indexVK] += V[indexV] * K[indexK];
    }
}




template<typename T>
__global__ void matrix_multiply_kernel(
    const T* Q, const T* VK, T* output,
    int N, int H, int S, int d_V, int d_K, int s) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d_v = threadIdx.x; // Dimension index within d_V (output dimension)

    // Ensure we are within bounds for the d_V dimension
    if (d_v < d_V) {
        T sum = 0;
        // Perform the dot product along the D dimension (which corresponds to d_K in VK)
        for (int D_idx = 0; D_idx < d_K; ++D_idx) {
            // Compute indices for Q and VK. Note that VK does not vary with s,
            // so we use a fixed sequence index (effectively 0) for VK.
            int indexQ = ((n * H + h) * S + s) * d_K + D_idx;
            // For VK, since it's (N, H, 1, d_V, d_K), we don't include 's' in its index calculation
            int indexVK = (((n * H + h) * 1 + 0) * d_V + d_v) * d_K + D_idx;

            // Perform element-wise multiplication and accumulate the sum
            sum += Q[indexQ] * VK[indexVK];
        }

        // Write the accumulated sum to the output tensor
        int indexOutput = ((n * H + h) * S + s) * d_V + d_v;
        output[indexOutput] = sum;
    }
}




template<typename T>
void compute_attention(
    const T* Q, const T* K, const T* V,
    T* output,
    int N, int H, int S, int d_V, int d_K,
    cudaStream_t stream = 0) {
    // Fixed number of threads per block
    int threadsPerBlock = 1024;

    // One block per batch-dimension index and head-dimension index
    dim3 grid(N, H);

    // Intermediate tensor to store the product between V and K
    // at each position in the sequence
    // The shape is (N, H, 1, d_V, d_K)
    T* VK;
    // Initialize to zeros
    cudaMalloc(&VK, N * H * 1 * d_V * d_K * sizeof(T));
    cudaMemset(VK, 0, N * H * 1 * d_V * d_K * sizeof(T));

    // writeTensorToFile("Q.bin", Q, {N, H, S, d_K});
    // writeTensorToFile("K.bin", K, {N, H, S, d_K});
    // writeTensorToFile("V.bin", V, {N, H, S, d_V});

    // Iterate over the sequence dimension and compute the outer product
    for (int s = 0; s < S; ++s) {
        // Launch the kernel
        // This will compute the outer product between V and K at each position in the sequence
        // and add the result to VK
        compute_outer_product_kernel<T><<<grid, threadsPerBlock, 0, stream>>>(K, V, VK, N, H, S, d_V, d_K, s);

        // Wait for the kernel to complete
        cudaDeviceSynchronize();

        // Product between Q at position s and VK
        // This is the output for the s-th position in the sequence
        matrix_multiply_kernel<T><<<grid, d_V, 0, stream>>>(Q, VK, output, N, H, S, d_V, d_K, s);

        // Wait for the kernel to complete
        cudaDeviceSynchronize();
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
    int threadsPerBlock = 256; // Example, adjust based on device capabilities
    int blocksPerGrid = (N * H * S * D + threadsPerBlock - 1) / threadsPerBlock;

    compute_attention<T>(Q, K, V, output, N, H, S, D, D, stream);
}





// // CUDA forward declarations
// void compute_and_contract_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);
// void compute_and_contract(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);

// C++ interface
void compute_and_contract_call(const torch::Tensor& Q, const torch::Tensor& K_orig, const torch::Tensor& V_orig, torch::Tensor& output) {
    // Check tensor requirements, e.g., dtype, device, etc.
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K_orig.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V_orig.device().is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

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




// Debugging
#include <iostream>
#include <chrono>
// dummy main function
int main() {
    // Set the device
    torch::Device device(torch::kCUDA, 0);

    // Set the tensor dimensions
    int N = 16;
    int H = 8;
    int S = 64;
    int D = 32;

    // Create input tensors
    auto Q = torch::rand({N, H, S, D}, device);
    auto K = torch::rand({N, H, S, D}, device);
    auto V = torch::rand({N, H, S, D}, device);

    // Create output tensor
    auto output = torch::zeros({N, H, S, D}, device);

    // Call the custom CUDA kernel
    auto start = std::chrono::high_resolution_clock::now();
    compute_and_contract_call(Q, K, V, output);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    return 0;
}
