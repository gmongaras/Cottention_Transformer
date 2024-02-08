// #include <cuda_runtime.h>
// #include <cublas_v2.h>
// #include <torch/extension.h>

// // template<typename T>
// // void compute_and_contract_cuda_impl(const T* A, const T* B, const T* C, T* output, int N, int H, int S, int D);

// // // Define the wrapper function to call the kernel
// // void compute_and_contract_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output) {
// //     // Extract tensor dimensions and call the template function
// //     auto N = A.size(0);
// //     auto H = A.size(1);
// //     auto S = A.size(2);
// //     auto D = A.size(3);

// //     AT_DISPATCH_FLOATING_TYPES(A.type(), "compute_and_contract_cuda", ([&] {
// //         compute_and_contract_cuda_impl<scalar_t>(
// //             A.data_ptr<scalar_t>(),
// //             B.data_ptr<scalar_t>(),
// //             C.data_ptr<scalar_t>(),
// //             output.data_ptr<scalar_t>(),
// //             N, H, S, D);
// //     }));
// // }

// template<typename T>
// __global__ void compute_and_contract_kernel(
//     const T* __restrict__ A, const T* __restrict__ B, const T* __restrict__ C,
//     T* __restrict__ output, int N, int H, int S, int D) {
//     extern __shared__ T sharedMem[];

//     int b = blockIdx.x; // Batch index
//     int h = blockIdx.y; // Head index
//     int s = blockIdx.z; // Sequence index
//     int d = threadIdx.x; // Dimension index

//     if (b < N && h < H && s < S && d < D) {
//         // Compute cumulative product directly
//         T cumProd = 0;
//         for (int k = 0; k <= d; ++k) {
//             int idxA = b * H * S * D + h * S * D + s * D + k;
//             int idxB = b * H * S * D + h * S * D + k * S + s; // Transpose S and D for B
//             cumProd += A[idxA] * B[idxB];
//         }

//         // Store cumulative product in shared memory for reduction
//         sharedMem[threadIdx.x] = cumProd;
//         __syncthreads();

//         // Reduction within a block to compute the final output
//         // Assuming D is a power of 2 for simplicity. For non-power of 2, additional handling is needed.
//         for (int stride = D / 2; stride > 0; stride >>= 1) {
//             if (d < stride) {
//                 sharedMem[d] += sharedMem[d + stride];
//             }
//             __syncthreads();
//         }

//         // Use the result of reduction to compute the final tensor contraction with C
//         if (d == 0) {
//             int idxC = b * H * S * D + h * S * D + s * D; // Index for C
//             int idxOutput = b * H * S * D + h * S * D + s * D; // Index for output
//             output[idxOutput] = sharedMem[0] * C[idxC];
//         }
//     }
// }

// template<typename T>
// void compute_and_contract(
//     const T* A, const T* B, const T* C, T* output,
//     int N, int H, int S, int D,
//     cudaStream_t stream = 0) {
//     dim3 grid(N, H, S);
//     int threads = D;
//     int sharedMemSize = D * sizeof(T);
//     compute_and_contract_kernel<<<grid, threads, sharedMemSize, stream>>>(A, B, C, output, N, H, S, D);
// }


#include <torch/torch.h>
// #include <torch/extension.h>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>




//// Used to do (K.unsqueeze(-1)*V.unsqueeze(-2))
template<typename T>
__global__ void compute_outer_product_kernel(
    const T* __restrict__ K, const T* __restrict__ V, T* __restrict__ KV,
    int N, int H, int S, int D) {
    int b = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int s = blockIdx.z; // Sequence index
    int d1 = threadIdx.x; // Dimension index for K
    int d2 = threadIdx.y; // Dimension index for V

    if (b < N && h < H && s < S && d1 < D && d2 < D) {
        // Get indices for K, V, and KV
        int idxK = b * H * S * D + h * S * D + s * D + d1;
        int idxV = b * H * S * D + h * S * D + s * D + d2;
        int idxKV = b * H * S * D * D + h * S * D * D + s * D * D + d1 * D + d2;

        // Perform the outer product operation
        T k_val = K[idxK];
        T v_val = V[idxV];
        KV[idxKV] = k_val * v_val; // Element-wise multiplication for outer product
    }
}
template<typename T>
void compute_outer_product(
    const T* K, const T* V, T* KV,
    int N, int H, int S, int D,
    cudaStream_t stream = 0) {
    dim3 grid(N, H, S);
    int threads = D;
    compute_outer_product_kernel<<<grid, threads, 0, stream>>>(K, V, KV, N, H, S, D);
}




// Used to do (K.unsqueeze(-1)*V.unsqueeze(-2)).cumsum(2)
template<typename T>
__global__ void compute_cumsum_over_S_kernel(
    T* __restrict__ KV, // KV is both input and output
    int N, int H, int S, int D) {
    int b = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int d1 = blockIdx.z / D; // Dimension index for the first D
    int d2 = blockIdx.z % D; // Dimension index for the second D
    int s = threadIdx.x; // Sequence index, used for cumsum

    if (b < N && h < H && d1 < D && d2 < D && s < S) {
        int baseIdx = b * H * S * D * D + h * S * D * D + d1 * D * D + d2;
        T sum = 0;
        for (int seq = 0; seq <= s; ++seq) {
            int idx = baseIdx + seq * D * D;
            sum += KV[idx];
            KV[idx] = sum; // Store the cumsum back into KV
        }
    }
}






// Compute the product of VK with Q: torch.einsum("bsD,bsdD->bsd", Q, VK)
template<typename T>
__global__ void matrix_vector_multiply_sum_kernel(
    const T* __restrict__ Q, const T* __restrict__ VK, T* __restrict__ output,
    int N, int H, int S, int D) {
    int b = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int s = blockIdx.z; // Sequence index
    int d = threadIdx.x; // Dimension index in the output and Q

    if (b < N && h < H && s < S && d < D) {
        T sum = 0;
        int idxQ = b * H * S * D + h * S * D + s * D + d;
        int baseIdxVK = b * H * S * D * D + h * S * D * D + s * D * D + d;
        
        for (int i = 0; i < D; ++i) {
            int idxVK = baseIdxVK + i * D; // Move across the last D dimension in VK
            sum += Q[idxQ] * VK[idxVK];
        }
        int idxOutput = b * H * S * D + h * S * D + s * D + d;
        output[idxOutput] = sum;
    }
}







// Wrapper function to orchestrate the computation
template<typename T>
void compute_and_contract(
    const T* Q, const T* K, const T* V, T* output,
    int N, int H, int S, int D,
    cudaStream_t stream = 0) {
    T* KV;
    cudaMalloc(&KV, N * H * S * D * D * sizeof(T)); // Allocate space for KV

    // Compute the outer product KV = K.unsqueeze(-1) * V.unsqueeze(-2)
    compute_outer_product(K, V, KV, N, H, S, D, stream);


    // Compute the cumsum over S for KV: (K.unsqueeze(-1) * V.unsqueeze(-2)).cumsum(2)
    // Calculate the number of blocks for the cumsum kernel
    dim3 gridCumsum(N, H, D * D);
    int threadsCumsum = S;
    // Make sure S does not exceed the maximum number of threads per block
    // If S is larger, you'll need to adjust the strategy for cumsum calculation
    compute_cumsum_over_S_kernel<<<gridCumsum, threadsCumsum, 0, stream>>>(KV, N, H, S, D);



    // Compute the product of VK with Q: torch.einsum("bsD,bsdD->bsd", Q, VK)
    // Calculate matrix-vector multiplication and summation
    dim3 gridMulti(N, H, S);
    int threadsMulti = D;
    matrix_vector_multiply_sum_kernel<<<gridMulti, threadsMulti, 0, stream>>>(Q, KV, output, N, H, S, D);



    cudaFree(KV);
}



// // CUDA forward declarations
// void compute_and_contract_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);
// void compute_and_contract(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);

// C++ interface
void compute_and_contract_call(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V, torch::Tensor& output) {
    // Check tensor requirements, e.g., dtype, device, etc.
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    // Get tensor dimensions
    auto N = Q.size(0);
    auto H = Q.size(1);
    auto S = Q.size(2);
    auto D = Q.size(3);

    // Call the CUDA kernel
    compute_and_contract<float>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        output.data_ptr<float>(),
        N, H, S, D);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("compute_and_contract", &compute_and_contract_call, "Compute and contract operation");
// }




// Debugging
#include <iostream>
#include <chrono>
// dummy main function
int main() {
    // Set the device
    torch::Device device(torch::kCUDA, 0);

    // Set the tensor dimensions
    int N = 32;
    int H = 12;
    int S = 64;
    int D = 128;

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
