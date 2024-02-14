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
    int N, int H, int S, int d_V, int d_K) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int s = blockIdx.z; // Sequence index

    // Each thread handles a portion of the elements in VK
    int threadsPerBlock = blockDim.x;

    // Calculate the number of elements in VK
    int numElements = d_V * d_K;

    // Calculate the number of elements per thread.
    int elementsPerThread = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Calculate the start and end indices for the thread
    int startIdx = threadIdx.x * elementsPerThread;

    // Ensure the end index does not exceed the number of elements
    int endIdx = min(startIdx + elementsPerThread, numElements);

    // Loop over the elements this thread is responsible for
    for (int idx = startIdx; idx < endIdx; ++idx) {
        int d_v = idx / d_V; // Dimension index for V
        int d_k = idx % d_V; // Dimension index for K

        // Ensure we are within bounds since d_k and d_v are derived from threadIdx.x
        if (d_k < d_K && d_v < d_V) {
            // Compute indices for V, K, and VK
            // Index is as follows: (N, H, S, ...) -> ((n * H + h) * S + s) * ...
            int indexV = ((n * H + h) * S + s) * d_V + d_v; // (N, H, S, d_V)
            int indexK = ((n * H + h) * S + s) * d_K + d_k; // (N, H, S, d_K)
            int indexVK = ((((n * H + h) * S + s) * d_V) + d_v) * d_K + d_k; // (N, H, S, d_V, d_K)

            // Value at (N, H, s, d_V, d_K) = (N, H, s, d_V) * (N, H, s, d_K)
            // where d is varies
            VK[indexVK] = V[indexV] * K[indexK];
        }
    }
}
template<typename T>
void compute_outer_product(
    const T* K, const T* V, T* VK,
    int N, int H, int S, int d_V, int d_K,
    cudaStream_t stream = 0) {
    // Fixed number of threads per block
    int threadsPerBlock = 1024;

    dim3 grid(N, H, S);

    // Launch the outer product kernel
    compute_outer_product_kernel<T><<<grid, threadsPerBlock, 0, stream>>>(K, V, VK, N, H, S, d_V, d_K);

    // Ensure kernel execution completes before copying data
    cudaDeviceSynchronize();
}











// Used to do (V.unsqueeze(-1)*K.unsqueeze(-2)).cumsum(2)
template<typename T>
__global__ void cumsum_over_s_kernel(
    T* VK, int N, int H, int S, int d_K, int d_V) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d_v = threadIdx.x; // Assuming d_K <= max threads per block
    int d_k = threadIdx.y; // Assuming d_V <= max threads per block

    // Ensure we are within bounds since d_k and d_v are derived from threadIdx.x and threadIdx.y
    if (d_k < d_K && d_v < d_V) {
        // Loop over the sequence dimension to compute the cumulative sum
        for (int s = 1; s < S; ++s) {
            int idx = ((((n * H + h) * S + s) * d_V + d_v) * d_K + d_k);
            int prev_idx = ((((n * H + h) * S + (s - 1)) * d_V + d_v) * d_K + d_k);
            VK[idx] += VK[prev_idx];
        }
    }
}
template<typename T>
void cumsum_over_s(
    T* VK, int N, int H, int S, int d_V, int d_K,
    cudaStream_t stream = 0) {
    // Define the number of blocks and threads
    dim3 blocks(N, H);
    dim3 threads(d_V, d_K); // Ensure d_K and d_V do not exceed the maximum threads per block

    // Launch the kernel
    cumsum_over_s_kernel<T><<<blocks, threads, 0, stream>>>(VK, N, H, S, d_V, d_K);

    // Optionally synchronize
    cudaDeviceSynchronize();
}









// Compute the product of VK with Q: torch.einsum("bhsD,bhsdD->bhsd", Q, VK)
template<typename T>
__global__ void dot_product_over_D_kernel(
    const T* Q, const T* VK, T* output,
    int N, int H, int S, int d, int D) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int s = blockIdx.z / d; // Sequence index
    int d_idx = blockIdx.z % d; // Dimension index for d

    if (d_idx < d && s < S && h < H && n < N) {
        T sum = 0;
        for (int D_idx = 0; D_idx < D; ++D_idx) {
            int Q_index = ((n * H + h) * S + s) * D + D_idx;
            int VK_index = ((((n * H + h) * S + s) * d + d_idx) * D + D_idx);
            sum += Q[Q_index] * VK[VK_index];
        }
        int output_index = (((n * H + h) * S + s) * d + d_idx);
        output[output_index] = sum;
    }
}
template<typename T>
void dot_product_over_D(
    const T* Q, const T* VK, T* output,
    int N, int H, int S, int d, int D,
    cudaStream_t stream = 0) {

    // Calculate the total number of elements in the output tensor
    int totalElements = N * H * S * d;
    
    // Define the number of blocks and threads
    // Assuming each thread block handles one (n, h, s, d) combination
    int threadsPerBlock = 1; // Single thread per combination for simplicity
    int numBlocks = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
    
    // Adjusted to launch one block for each (n, h, s, d) combination explicitly
    dim3 blocks(N, H, S * d);

    // Launch the kernel
    dot_product_over_D_kernel<T><<<blocks, threadsPerBlock, 0, stream>>>(Q, VK, output, N, H, S, d, D);

    // Optionally synchronize (or handle it outside this function)
    cudaDeviceSynchronize();
}









// Wrapper function to orchestrate the computation
template<typename T>
void compute_and_contract(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    cudaStream_t stream = 0) {
    T* VK;
    cudaMalloc(&VK, N * H * S * D * D * sizeof(T)); // Allocate space for VK

    // Compute the outer product VK = V.unsqueeze(-1) * K.unsqueeze(-2)
    compute_outer_product(K, V, VK, N, H, S, D, D, stream);


    // Compute the cumsum over S for VK: (V.unsqueeze(-1) * K.unsqueeze(-2)).cumsum(2)
    cumsum_over_s(VK, N, H, S, D, D, stream);



    // Compute the product of VK with Q: torch.einsum("bhsD,bhsdD->bhsd", Q, VK)
    dot_product_over_D(Q, VK, output, N, H, S, D, D, stream);


    cudaFree(VK);
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
