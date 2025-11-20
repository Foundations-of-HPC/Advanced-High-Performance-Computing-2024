#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printNumber(int stream_id, int number)
{
    printf("Stream %d -> %d\n", stream_id, number);
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
    const int NUM_STREAMS = 5;

    // Serial part: no streams, run each kernel sequentially
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        printNumber<<<1, 1>>>(i, i * 10);

        // Ensure each kernel finishes before the next one starts
        CHECK(cudaDeviceSynchronize());
    }

    return 0;
}

