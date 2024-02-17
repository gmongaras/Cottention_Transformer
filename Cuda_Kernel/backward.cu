#include <cuda_runtime.h> // For cudaMemcpy and cudaFree
#include <torch/torch.h>
// #include <torch/extension.h>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <chrono>

#include <cuda_fp16.h> // Include CUDA half-precision definitions




// // General AtomicAdd_
// template<typename T>
// __device__ void AtomicAdd_(T* address, T val) {
//     atomicAdd(address, val);
// }
// // Specialization for half precision
// template<>
// __device__ void AtomicAdd_(at::Half* address, at::Half val) {
//     atomicAdd(reinterpret_cast<__half*>(address), *reinterpret_cast<__half*>(&val));
// }
// // Specialization for bfloat16 half precision
// template<>
// __device__ void AtomicAdd_(at::BFloat16* address, at::BFloat16 val) {
//     atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), *reinterpret_cast<__nv_bfloat16*>(&val));
// }




// // For debugging
// #include <fstream>
// template<typename T>
// void writeTensorToFile(const std::string& filename, const T* tensorData, const std::vector<int>& shape) {
//     std::ofstream file(filename, std::ios::binary | std::ios::out);

//     // Write the shape
//     int dimensions = shape.size();
//     file.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
//     for (int dim : shape) {
//         file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
//     }

//     // Get the number of elements in the tensor
//     int numElements = 1;
//     for (int dim : shape) numElements *= dim;

//     // Allocate host memory to copy the tensor data
//     size_t numBytes = numElements * sizeof(T);
//     T* hostData = new T[numElements];

//     // Copy the tensor data to host
//     cudaMemcpy(hostData, tensorData, numBytes, cudaMemcpyDeviceToHost);

//     // Write the tensor data to file
//     file.write(reinterpret_cast<const char*>(hostData), sizeof(T) * numElements);

//     // Close the file and free the host memory
//     file.close();
//     free(hostData);
// }






// Used to do an inplace sum over the d dimension
template<typename T>
__global__ void inplace_sum_over_d(
    T* tensor, int N, int H, int S, int D) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int s = blockIdx.z; // Sequence index

    // Get the sum of the tensor over the d dimension
    T sum = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sum += tensor[((n * H + h) * S + s) * D + d];
    }

    // Replace all values on the d dimension with the sum
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        tensor[((n * H + h) * S + s) * D + d] = sum;
    }
}







// Used to do Q.flip(over S).cumsum(over d).flip(over S)
template<typename T>
__global__ void cumsum_reversed(
    T* Q, int N, int H, int S, int D) {
    int n = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d = threadIdx.x; // Dimension index within the sequence

    // Allocate shared memory for the cumulative sum
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);

    // Ensure we are within bounds
    if (d < D) {
        // Iterate over the sequence dimension and compute the cumulative sum for each position
        for (int s = 0; s < S; s++) {
            // Store the value of Q at this position
            shared_memory[d] = Q[((n * H + h) * S + s) * D + d];

            // Sum all positions greater than this one
            for (int i = s+1; i < S; i++) {
                int indexQ = ((n * H + h) * S + i) * D + d;

                // Add the value to the shared memory
                shared_memory[d] += Q[indexQ];
            }

            // Wait for all threads to finish writing to shared memory
            __syncthreads();

            // Copy the contents of the shared memory to the output tensor
            for (int i = 0; i < D; i++) {
                int indexQ = ((n * H + h) * S + s) * D + i;
                Q[indexQ] = shared_memory[i];
            }

            // Wait for all threads to finish writing to the output tensor
            __syncthreads();
        }
    }
}



template<typename T>
void compute_backward(
    torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V,
    torch::Tensor& temp, torch::Tensor& previous_grad,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {
    // Grid for the matrix multiplication kernel
    // One block per batch-dimension index, head-dimension index, and both dimensions of VK

    // writeTensorToFile("Q.bin", Q, {N, H, S, d_K});
    // writeTensorToFile("K.bin", K, {N, H, S, d_K});
    // writeTensorToFile("V.bin", V, {N, H, S, d_V});


    // 1: compute the sum of V along the d dimension
    // V = V.sum(-1, true);
    dim3 grid_sum(N, H, S);
    inplace_sum_over_d<T><<<grid_sum, 1, 0, stream>>>(V.data_ptr<T>(), N, H, S, D);

    // 2: Kernel to compute the cumulative sum of Q over S reversed
    dim3 grid(N, H);
    cumsum_reversed<T><<<grid, D, D*sizeof(T), stream>>>(Q.data_ptr<T>(), N, H, S, D);

    // 3: Temp stores the gradient of Q - (V * K).cumsum(over S)
    // temp = (V.sum(over d) * K).cumsum(over S) * previous_grad;
    temp.mul_(V).mul_(K);
    temp = temp.cumsum(2);
    // cumsum_reversed<T><<<grid, D, D*sizeof(T), stream>>>(temp.data_ptr<T>(), N, H, S, D); // (V * K).cumsum(over S)
    temp.mul_(previous_grad); // (V * K).cumsum(over S) * previous_grad

    // 4: V stores the gradient of K - (V.sum(over d) * Q)
    // V = (V * Q) * previous_grad;
    V.mul_(Q).mul_(previous_grad);

    // 5:  K stores the gradient of V - (K * Q).sum(over d)
    // K = (K * Q).sum(-1, true) * previous_grad;
    K.mul_(Q); // K * Q
    inplace_sum_over_d<T><<<grid_sum, 1, 0, stream>>>(K.data_ptr<T>(), N, H, S, D); // (K * Q).sum(-1, true)
    K.mul_(previous_grad); // (K * Q).sum(-1, true) * previous_grad
}



// Wrapper function to orchestrate the computation
template<typename T>
void backward_call(
    torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V, torch::Tensor& temp, torch::Tensor& previous_grad,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {
    compute_backward<T>(Q, K, V, temp, previous_grad,N, H, S, D, block_size, stream);
}





// // CUDA forward declarations
// void compute_and_contract_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);
// void compute_and_contract(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);

// C++ interface
template<typename dtype_>
torch::Tensor backward_(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V, torch::Tensor& previous_grad) {
    // Check tensor requirements, e.g., dtype, device, etc.
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");

    // Get tensor dimensions
    int N = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    // Singel tmeporary tensor
    auto temp = torch::ones({N, H, S, D}, torch::TensorOptions().dtype(Q.scalar_type()).device(Q.device()));

    // Ensure the tensors are contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    // Call the CUDA kernel
    backward_call<dtype_>(
        Q,
        K,
        V,
        temp,
        previous_grad,
        N, H, S, D, 1);

    // Gradient of Q, K, V
    return temp;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float32", &backward_<float>);
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
