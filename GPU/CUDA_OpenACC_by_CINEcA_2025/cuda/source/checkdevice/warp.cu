#include <cuda_runtime.h>
#include <stdio.h>

// nvidia-smi -a -q -d CLOCK | fgrep -A 3 "Max Clocks" | fgrep "Memory"

__global__ void printThreads() {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x; // Global thread ID
    int warpId = threadId / 32; // Determine the warp ID

    // Print the thread information
    printf("Warp %d: thread %d\n", warpId, threadId);
}

int main() {
    const int numThreads = 128; // Total number of threads
    const int blockSize = 32;   // Number of threads per block
    const int numBlocks = (numThreads + blockSize - 1) / blockSize; // Calculate number of blocks

    // Launch the kernel
    printThreads<<<numBlocks, blockSize>>>();

    // Wait for the GPU to finish before accessing the results
    cudaDeviceSynchronize();

    return 0;
}

