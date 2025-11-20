#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printNumber(int number)
{
    printf("Kernel %d -> %d\n", number, number * 10);
}

#define CHECK(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    }

int main()
{
    const int NUM_KERNELS = 5;

    // Launch each kernel sequentially (no streams)
    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        printf("Launching kernel %d...\n", i);  // Print a message indicating kernel launch

        // Launch the kernel
        printNumber<<<1, 1>>>(i);
        
        // Synchronize the device to ensure kernel finishes before the next one starts
        CHECK(cudaDeviceSynchronize());
    }

    return 0;
}

