#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernels.cuh"
#include "utils.h"
#include "scan.cuh"

// Macro for CUDA error checking
#define checkCudaError(o, l) _checkCudaError(o, l, __func__)

// Constants for GPU execution
int THREADS_PER_BLOCK = 512; // Number of threads per block
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2; // Elements processed per block

/**
 * Performs a parallel prefix sum (scan) on an integer array.
 *
 * @param output Pointer to the output array where the scan results are stored.
 * @param input Pointer to the input array to be scanned.
 * @param length Number of elements in the input array.
 * @return Elapsed time in milliseconds to perform the scan operation.
 */
float scan(int *output, int *input, int length) {
    int *d_out, *d_in;
    const int arraySize = length * sizeof(int);

    // Allocate GPU memory for input and output arrays
    cudaMalloc((void **)&d_out, arraySize);
    cudaMalloc((void **)&d_in, arraySize);

    // Copy input data from host to device
    cudaMemcpy(d_out, output, arraySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, input, arraySize, cudaMemcpyHostToDevice);

    // Initialize timer events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Choose the appropriate scan function based on input size
    if (length > ELEMENTS_PER_BLOCK) {
        scanLargeDeviceArray(d_out, d_in, length);
    } else {
        scanSmallDeviceArray(d_out, d_in, length);
    }

    // Stop timer and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy the scan result back to host memory
    cudaMemcpy(output, d_out, arraySize, cudaMemcpyDeviceToHost);

    // Clean up GPU resources
    cudaFree(d_out);
    cudaFree(d_in);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}

/**
 * Handles the scan of large arrays by dividing the work into smaller chunks.
 *
 * @param d_out Device pointer to the output array.
 * @param d_in Device pointer to the input array.
 * @param length Number of elements in the input array.
 */
void scanLargeDeviceArray(int *d_out, int *d_in, int length) {
    int remainder = length % ELEMENTS_PER_BLOCK;
    if (remainder == 0) {
        scanLargeEvenDeviceArray(d_out, d_in, length);
    } else {
        // Handle arrays not evenly divisible by ELEMENTS_PER_BLOCK
        int lengthMultiple = length - remainder;
        scanLargeEvenDeviceArray(d_out, d_in, lengthMultiple);

        // Scan remaining elements separately
        int *startOfOutputArray = &(d_out[lengthMultiple]);
        scanSmallDeviceArray(startOfOutputArray, &(d_in[lengthMultiple]), remainder);

        // Adjust the final segment with the previous sum
        add<<<1, remainder>>>(startOfOutputArray, remainder, &(d_in[lengthMultiple - 1]), &(d_out[lengthMultiple - 1]));
    }
}

/**
 * Performs a scan operation on small arrays that fit within a single block.
 *
 * @param d_out Device pointer to the output array.
 * @param d_in Device pointer to the input array.
 * @param length Number of elements in the input array.
 */
void scanSmallDeviceArray(int *d_out, int *d_in, int length) {
    int powerOfTwo = nextPowerOfTwo(length);
    prescan_arbitrary << <1, (length + 1) / 2, 2 * powerOfTwo * sizeof(int)>> >(d_out, d_in, length, powerOfTwo);
}

/**
 * Executes the scan operation on large arrays that are evenly divisible by ELEMENTS_PER_BLOCK.
 *
 * @param d_out Device pointer to the output array.
 * @param d_in Device pointer to the input array.
 * @param length Number of elements in the input array.
 */
void scanLargeEvenDeviceArray(int *d_out, int *d_in, int length) {
    const int blocks = length / ELEMENTS_PER_BLOCK;
    const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

    // Allocate memory for intermediate sums and increments
    int *d_sums, *d_incr;
    cudaMalloc((void **)&d_sums, blocks * sizeof(int));
    cudaMalloc((void **)&d_incr, blocks * sizeof(int));

    // Execute the large prescan kernel
    prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(d_out, d_in, ELEMENTS_PER_BLOCK, d_sums);

    // Determine if a second level of scanning is needed for the block sums
    const int sumsArrThreadsNeeded = (blocks + 1) / 2;
    if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
        scanLargeDeviceArray(d_incr, d_sums, blocks);
    } else {
        scanSmallDeviceArray(d_incr, d_sums, blocks);
    }

    // Adjust output with block increments
    add<<<blocks, ELEMENTS_PER_BLOCK>>>(d_out, ELEMENTS_PER_BLOCK, d_incr);

    // Clean up allocated memory
    cudaFree(d_sums);
    cudaFree(d_incr);
}