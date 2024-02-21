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
void compute_backward_old(
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
void backward_call_old(
    torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V, torch::Tensor& temp, torch::Tensor& previous_grad,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {
    compute_backward_old<T>(Q, K, V, temp, previous_grad,N, H, S, D, block_size, stream);
}





// // CUDA forward declarations
// void compute_and_contract_cuda(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);
// void compute_and_contract(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, torch::Tensor& output);

// C++ interface
template<typename dtype_>
torch::Tensor backward_old(torch::Tensor& Q, torch::Tensor& K, torch::Tensor& V, torch::Tensor& previous_grad) {
    // Check tensor requirements, e.g., dtype, device, etc.
    TORCH_CHECK(Q.device().is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.device().is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.device().is_cuda(), "V must be a CUDA tensor");

    // Get tensor dimensions
    int N = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    // Single temporary tensor
    auto temp = torch::ones({N, H, S, D}, torch::TensorOptions().dtype(Q.scalar_type()).device(Q.device()));

    // Ensure the tensors are contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    temp = temp.contiguous();
    previous_grad = previous_grad.contiguous();

    // https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan.cpp
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)Q.get_device()};
    // c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Call the CUDA kernel
    backward_call_old<dtype_>(
        Q,
        K,
        V,
        temp,
        previous_grad,
        N, H, S, D, 1);
    // backward_call<dtype_>(
    //     at::autocast::cached_cast(at::kHalf, Q),
    //     at::autocast::cached_cast(at::kHalf, K),
    //     at::autocast::cached_cast(at::kHalf, V),
    //     at::autocast::cached_cast(at::kHalf, temp),
    //     at::autocast::cached_cast(at::kHalf, previous_grad),
    //     N, H, S, D, 1);

    // Gradient of Q, K, V
    return temp;
}











// Hehe
// ^w^
// UwU
// OwO
// Nyaa~ <- What I say when I'm coding in cuda. Nya~~~
// Rawr~
// >w<
// >_<
// >.<
// >:3 <- Doesn't matter if it's a cat or a dog. It's a catdog.
// >:D 
// >:P
// >:( <- This me. IDK how to code cuda kernels.
// ^w^
// UwU
// Nya~~







// Used to do (G.unsqueeze(-1)*Q.unsqueeze(-2)) for each position in the sequence
template<typename T>
__global__ void compute_outer_products(
    const T* G, const T* Q, T* GQ,
    int N, int H, int S, int d_G, int d_Q, int s, int block_size, int BS) {
    int n = floor((float)blockIdx.x / (float)H); // Batch index
    int h = blockIdx.x % H; // Head index
    int blk_idx = blockIdx.y; // Dimension index within the sequence
    int d_g = blockIdx.z; // Dimension index within d_G
    int d_q = threadIdx.x; // Dimension index within d_Q

    // Ensure we are within bounds for the d_G dimension and d_Q dimension
    if (d_g < d_G && d_q < d_Q) {
        // Compute indices for G and Q at the current block in the sequence
        int indexG = ((n * H + h) * S + s + blk_idx) * d_G + d_g;
        int indexQ = ((n * H + h) * S + s + blk_idx) * d_Q + d_q;

        // Do the outer product between V and K
        T product = G[indexG] * Q[indexQ];

        // Iterate over all blocks with a block index greater than this one
        // and add the product to the GQ tensor, thus doing a cumulative sum
        // in the shared memory.
        for (int i = blk_idx; i < BS; i++) {
            int indexGQ = (((n * H + h) * block_size + i) * d_G + d_g) * d_Q + d_q;

            // Add the product to the VK tensor
            AtomicAdd_(&GQ[indexGQ], product);
        }
    }
}




template<typename T>
__global__ void matrix_multiply_kernel(
    const T* V, const T* K, T* GQ,
    T* K_grad, T* V_grad,
    int N, int H, int S, int d_G, int d_Q, int s, int block_size, int BS) {
    int n = floor((float)blockIdx.x / (float)H); // Batch index
    int h = blockIdx.x % H; // Head index
    int blk_idx = blockIdx.y; // Dimension index within the sequence
    int d_g = blockIdx.z; // Dimension index within d_G
    int d_q = threadIdx.x; // Dimension index within d_Q


    // Allocate shared memory for the cumulative sum
    // My man!
    // https://github.com/pytorch/extension-cpp/issues/59#issuecomment-626189915
    extern __shared__ __align__(sizeof(T)) unsigned char shared_memory_uchar[];
    T *shared_memory = reinterpret_cast<T *>(shared_memory_uchar);

    // Ensure we are within bounds for the d_G dimension and d_Q dimension
    if (d_g < d_G && d_q < d_Q) {
        // Compute indices for V, K, and GQ. Note that GQ does not vary with s,
        // so we use a fixed sequence index (effectively 0) for GQ.
        // V is based on G (sum over d_G) and K is based on Q (sum over d_Q)
        int indexV = ((n * H + h) * S + s + blk_idx) * d_Q + d_q;
        int indexK = ((n * H + h) * S + s + blk_idx) * d_Q + d_q;

        // For GQ, since it's (N, H, 1, d_G, d_Q), we don't include 's' in its index calculation
        int indexGQ_K = (((n * H + h) * block_size + blk_idx) * d_G + d_g) * d_Q + d_q;
        int indexGQ_V = (((n * H + h) * block_size + blk_idx) * d_Q + d_q) * d_G + d_g; // For V it's flipped as we sum over d_Q as dim=-1


        // Multiply the V,K and GQ tensors and accumulate the sum in shared memory
        shared_memory[d_q] = K[indexK] * GQ[indexGQ_K];  // Use first half of memory for K
        shared_memory[d_Q + d_q] = V[indexV] * GQ[indexGQ_V]; // Use second half of memory for V

        // Wait for all threads to finish writing to shared memory
        __syncthreads();

        // Only one thread sums all the elements in shared memory and stores the result in the output tensors
        if (d_q == 0) {
            int indexOutput = ((n * H + h) * S + s + blk_idx) * d_G + d_g;

            T sum_K = 0;
            T sum_V = 0;
            for (int i = 0; i < d_Q; i++) {
                sum_K += shared_memory[i];
                sum_V += shared_memory[i + d_Q];
            }
            
            V_grad[indexOutput] = sum_K; // Derivative of V is based on K
            K_grad[indexOutput] = sum_V; // Derivative of K is based on V
        }

        // // Since each position in GQ is only access once, we can copy
        // // the contents of the last block to this one.
        // // This ensures the cumulative sum is correct for the next block.
        // // Only do this copy if the current block is not the last one
        // if (blk_idx < BS-1) {
        //     GQ[indexGQ] = VK[(((n * H + h) * block_size + BS-1) * d_G + d_g) * d_Q + d_q];
        // }
    }
}




template<typename T>
void compute_backward(
    const T* Q, const T* K, const T* V, const T* prev_grad,
    T* K_grad, T* V_grad, 
    T* GQ,
    int N, int H, int S, int d_G, int d_Q,
    const int block_size,
    cudaStream_t stream = 0) {

    // Blocks size greater than 1 not supported
    if (block_size > 1) {
        throw std::invalid_argument("Block size greater than 1 not supported. Final block not calculated if uneven");
    }

    // Iterate over the sequence dimension and compute the outer product
    // Note that this cumsum is reversed
    for (int s = S-1; s >= 0; s -= block_size) {
        // Block size cannot exceed the sequence length
        int BS = 1; //min(block_size, S-s);

        // Compute the cumulative product between G and Q
        // up to block_size positions in the sequence
        //   Grid over N, H, and grad dimension. Assuming the block size is small
        //      we can use this as the x index in the thread and y as the d_Q index
        //   Threads over the number of blocks and the Q dimension
        //   No shared memory
        //   Stream is the CUDA stream where the kernel will be executed
        dim3 grid(N*H, BS, d_G);
        compute_outer_products<T><<<grid, d_Q, 0, stream>>>(prev_grad, Q, GQ, N, H, S, d_G, d_Q, s, block_size, BS);

        // // Wait for the kernel to complete
        // cudaDeviceSynchronize();

        // Product between  both (K_grad - V at position s and GQ over dim d_G) and (V_grad - K at position s and GQ over dim d_Q)
        //   Grid over N, H, and value dimension. Assuming the block size is small
        //      we can use this as the x index in the thread and y as the d_G index
        //   Threads over the number of blocks and the d_G dimension
        //   Shared memory is used to accumulate the sum for both K_grad and V_grad
        //   stream - This is the CUDA stream where the kernel will be executed
        matrix_multiply_kernel<T><<<grid, d_Q, d_Q*2*sizeof(T), stream>>>(V, K, GQ, K_grad, V_grad, N, H, S, d_G, d_Q, s, block_size, BS);

        // // Wait for the kernel to complete
        // cudaDeviceSynchronize();
    }

    // writeTensorToFile("VK.bin", VK, {N, H, 1, d_V, d_K});
    // writeTensorToFile("output.bin", output, {N, H, S, d_V});
}



// Wrapper function to orchestrate the computation
template<typename T>
void backward_call(
    const T* Q, const T* K, const T* V, const T* previous_grad,
    T* K_grad, T* V_grad, 
    T* GQ,
    int N, int H, int S, int D,
    const int block_size,
    cudaStream_t stream = 0) {
    compute_backward<T>(Q, K, V, previous_grad, K_grad, V_grad, GQ, N, H, S, D, D, block_size, stream);
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

    // Ouput tensors, gradient of K and V
    auto K_grad = torch::zeros({N, H, S, D}, torch::TensorOptions().dtype(Q.scalar_type()).device(Q.device()));
    auto V_grad = torch::zeros({N, H, S, D}, torch::TensorOptions().dtype(Q.scalar_type()).device(Q.device()));

    // Allocate memory for the intermediate tensor - grad and Q outer product (without sum)
    auto GQ = torch::zeros({N, H, block_size, D, D}, torch::TensorOptions().dtype(Q.scalar_type()).device(Q.device()));

    // writeTensorToFile("Q.bin", Q.data_ptr<float>(), {N, H, S, D});
    // writeTensorToFile("K.bin", K_orig.data_ptr<float>(), {N, H, S, D});
    // writeTensorToFile("V.bin", V_orig.data_ptr<float>(), {N, H, S, D});

    // // Unsqueeze prev_grad along the last dimension and Q along the second-to-last dimension
    // auto prev_grad = prev_grad_orig.unsqueeze(-1); // (N, H, S, D, 1)
    // auto V = V_orig.unsqueeze(-2); // (N, H, S, 1, D)
    // Unsqueeze not needed as I am making the kernel hehe UwU

    // Ensure the tensors are contiguous
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();
    prev_grad = prev_grad.contiguous();

    // c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan.cpp
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)Q.get_device()};
    // c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);

    // Call the CUDA kernel
    backward_call<dtype_>(
        Q.data_ptr<dtype_>(),
        K.data_ptr<dtype_>(),
        V.data_ptr<dtype_>(),
        prev_grad.data_ptr<dtype_>(),
        K_grad.data_ptr<dtype_>(),
        V_grad.data_ptr<dtype_>(),
        GQ.data_ptr<dtype_>(),
        N, H, S, D, block_size);
    // forward_call<dtype_>(
    //     at::autocast::cached_cast(at::kHalf, Q).data_ptr<dtype_>(),
    //     at::autocast::cached_cast(at::kHalf, K).data_ptr<dtype_>(),
    //     at::autocast::cached_cast(at::kHalf, V).data_ptr<dtype_>(),
    //     at::autocast::cached_cast(at::kHalf, output).data_ptr<dtype_>(),
    //     at::autocast::cached_cast(at::kHalf, VK).data_ptr<dtype_>(),
    //     N, H, S, D, block_size);

    // writeTensorToFile("output.bin", output.data_ptr<float>(), {N, H, S, D});

    // return K_grad, V_grad;
    return {K_grad, V_grad};
}



// TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
//     m.def("float32", backward_<float>);
//     m.def("float16", backward_<at::Half>);
//     try {
//         m.def("bfloat16", backward_<at::BFloat16>);
//     } catch (const std::exception& e) {
//         std::cout << "GPU does not support bfloat16. Skipping..." << std::endl;
//         // std::cerr << "Error: " << e.what() << std::endl;
//     }
// }

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
