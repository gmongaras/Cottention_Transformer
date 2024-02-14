#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <torch/extension.h>

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

template<typename T>
__global__ void compute_and_contract_kernel(
    const T* __restrict__ A, const T* __restrict__ B, const T* __restrict__ C,
    T* __restrict__ output, int N, int H, int S, int D) {
    extern __shared__ T sharedMem[];

    int b = blockIdx.x; // Batch index
    int h = blockIdx.y; // Head index
    int s = blockIdx.z; // Sequence index
    int d = threadIdx.x; // Dimension index

    if (b < N && h < H && s < S && d < D) {
        // Compute cumulative product directly
        T cumProd = 0;
        for (int k = 0; k <= d; ++k) {
            int idxA = b * H * S * D + h * S * D + s * D + k;
            int idxB = b * H * S * D + h * S * D + k * S + s; // Transpose S and D for B
            cumProd += A[idxA] * B[idxB];
        }

        // Store cumulative product in shared memory for reduction
        sharedMem[threadIdx.x] = cumProd;
        __syncthreads();

        // Reduction within a block to compute the final output
        // Assuming D is a power of 2 for simplicity. For non-power of 2, additional handling is needed.
        for (int stride = D / 2; stride > 0; stride >>= 1) {
            if (d < stride) {
                sharedMem[d] += sharedMem[d + stride];
            }
            __syncthreads();
        }

        // Use the result of reduction to compute the final tensor contraction with C
        if (d == 0) {
            int idxC = b * H * S * D + h * S * D + s * D; // Index for C
            int idxOutput = b * H * S * D + h * S * D + s * D; // Index for output
            output[idxOutput] = sharedMem[0] * C[idxC];
        }
    }
}

template<typename T>
void compute_and_contract(
    const T* A, const T* B, const T* C, T* output,
    int N, int H, int S, int D,
    cudaStream_t stream = 0) {
    dim3 grid(N, H, S);
    int threads = D;
    int sharedMemSize = D * sizeof(T);
    compute_and_contract_kernel<<<grid, threads, sharedMemSize, stream>>>(A, B, C, output, N, H, S, D);
}


// #include <cuda_runtime.h>
// #include <cublas_v2.h>

// template<typename T>
// __global__ void compute_cumulative_product_kernel(
//     const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ AB,
//     int N, int H, int S, int D) {
//     int b = blockIdx.x; // Batch index
//     int h = blockIdx.y; // Head index
//     int s = blockIdx.z; // Sequence index
//     int d = threadIdx.x; // Dimension index

//     if (b < N && h < H && s < S && d < D) {
//         int idxA = b * H * S * D + h * S * D + s * D + d;
//         int idxB = b * H * S * D + h * S * D + d * S + s; // Transpose S and D for B
//         int idxAB = b * H * S * D * D + h * S * D * D + s * D * D + d;

//         T a_val = A[idxA];
//         T b_val = B[idxB];
//         T cumsum = 0;

//         for (int i = 0; i <= d; ++i) {
//             cumsum += a_val * b_val; // Element-wise multiplication
//             AB[idxAB + i * D] = cumsum; // Store cumulative sum
//         }
//     }
// }

// template<typename T>
// void compute_cumulative_product(
//     const T* A, const T* B, T* AB,
//     int N, int H, int S, int D,
//     cudaStream_t stream = 0) {
//     dim3 grid(N, H, S);
//     int threads = D;
//     compute_cumulative_product_kernel<<<grid, threads, 0, stream>>>(A, B, AB, N, H, S, D);
// }

// template<typename T>
// void tensor_contraction(
//     const T* C, const T* AB, T* output,
//     int N, int H, int S, int D,
//     cudaStream_t stream = 0) {
//     // Assuming C and AB are appropriately reshaped and transposed for multiplication,
//     // this pseudo code represents a high-level approach. Actual implementation might
//     // require using cuBLAS or manual kernel for optimized performance.

//     // Use cuBLAS for tensor contraction. This is a simplified example. In practice,
//     // you might need to set up leading dimensions and strides properly.
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     cublasSetStream(handle, stream);

//     const T alpha = 1;
//     const T beta = 0;

//     // For simplicity, assuming C is (N*H*S, D) and AB is (N*H*S, D, D),
//     // and we want to perform batched matrix multiplication.
//     // Adjust according to actual layout and dimensions.

//     // Setup descriptors for batched GEMM if using cuBLAS or perform manual
//     // multiplication in a custom kernel for the contraction.

//     cublasDestroy(handle);
// }

// // Wrapper function to orchestrate the computation
// template<typename T>
// void compute_and_contract(
//     const T* A, const T* B, const T* C, T* output,
//     int N, int H, int S, int D,
//     cudaStream_t stream = 0) {
//     T* AB;
//     cudaMalloc(&AB, N * H * S * D * D * sizeof(T)); // Allocate space for AB

//     compute_cumulative_product(A, B, AB, N, H, S, D, stream);
//     tensor_contraction(C, AB, output, N, H, S, D, stream);

//     cudaFree(AB);
// }


#include <torch/extension.h>
#include <vector>

// // CUDA forward declarations
// void compute_and_contract_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);
// void compute_and_contract(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);

// C++ interface
void compute_and_contract_call(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output) {
    // Check tensor requirements, e.g., dtype, device, etc.
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.device().is_cuda(), "C must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    // Get tensor dimensions
    auto N = A.size(0);
    auto H = A.size(1);
    auto S = A.size(2);
    auto D = A.size(3);

    // Call the CUDA kernel
    compute_and_contract<float>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        output.data_ptr<float>(),
        N, H, S, D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_and_contract", &compute_and_contract_call, "Compute and contract operation");
}
