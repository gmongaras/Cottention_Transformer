#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <torch/extension.h>

template<typename T>
void compute_and_contract_cuda_impl(const T* A, const T* B, const T* C, T* output, int N, int H, int S, int D);

// Define the wrapper function to call the kernel
void compute_and_contract_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output) {
    // Extract tensor dimensions and call the template function
    auto N = A.size(0);
    auto H = A.size(1);
    auto S = A.size(2);
    auto D = A.size(3);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "compute_and_contract_cuda", ([&] {
        compute_and_contract_cuda_impl<scalar_t>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, H, S, D);
    }));
}

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