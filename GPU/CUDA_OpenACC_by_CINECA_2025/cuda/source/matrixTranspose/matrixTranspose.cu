#include <stdio.h>
#include <cuda_runtime.h>
#include "helper.cuh"

#define BLOCK_SIZE 16
#define UNROLL_FACTOR 4

// Structure to hold kernel information
typedef struct {
    void (*func)(float*, float*, const int, const int);
    const char* name;
    dim3 gridDim;
} KernelInfo;

// Function to initialize kernel information
KernelInfo initializeKernel(int choice, int nx, int ny) {
    KernelInfo info;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    switch(choice) {
        case 0:
            info.func = copyRow;
            info.name = "CopyRow";
            break;
        case 1:
            info.func = copyCol;
            info.name = "CopyCol";
            break;
        case 2:
            info.func = transposeNaiveRow;
            info.name = "NaiveRow";
            break;
        case 3:
            info.func = transposeNaiveCol;
            info.name = "NaiveCol";
            break;
        case 4:
            info.func = transposeUnroll4Row;
            info.name = "Unroll4Row";
            grid.x = (nx + block.x * UNROLL_FACTOR - 1) / (block.x * UNROLL_FACTOR);
            break;
        case 5:
            info.func = transposeUnroll4Col;
            info.name = "Unroll4Col";
            grid.x = (nx + block.x * UNROLL_FACTOR - 1) / (block.x * UNROLL_FACTOR);
            break;
        default:
            printf("Invalid kernel choice. Using CopyRow.\n");
            info.func = copyRow;
            info.name = "CopyRow";
    }
    info.gridDim = grid;
    return info;
}

// Function to run the kernel and measure performance
float runKernel(KernelInfo info, float* d_out, float* d_in, int nx, int ny) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    cudaEventRecord(start);
    info.func<<<info.gridDim, block>>>(d_out, d_in, nx, ny);
    cudaEventRecord(stop);
    cudaCheckError();

    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed;
}

// Function to print results
void printResults(const char* kernelName, float bandwidth, float peakBandwidth) {
    printf("Effective Bandwidth of Kernels (L1 Cache Enabled)\n");
    printf("KERNEL                        BANDWIDTH        RATIO TO PEAK BANDWIDTH\n");
    printf("Theoretical peak bandwidth    %.1f GB/s\n", peakBandwidth);
    printf("%s : %.2f GB/s        %.2f%%\n", 
           kernelName, bandwidth, (bandwidth / peakBandwidth) * 100);
}

int main(int argc, char **argv) {
    const int nx = 16384;
    const int ny = 16384;
    const size_t nBytes = nx * ny * sizeof(float);
    const float theoretical_peak_bandwidth = 900.0f; // GB/s

    int kernelChoice = (argc > 1) ? atoi(argv[1]) : 0;

    // Allocate and initialize host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    if (h_A == nullptr || h_B == nullptr) {
        printf("Failed to allocate host memory\n");
        return -1;
    }
    initializeData(h_A, nx * ny);

    // Allocate and initialize device memory
    float *d_A, *d_B;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaCheckError();
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaCheckError();

    // Initialize and run kernel
    KernelInfo kernelInfo = initializeKernel(kernelChoice, nx, ny);
    float elapsed = runKernel(kernelInfo, d_B, d_A, nx, ny);

    // Calculate and print results
    float bandwidth = calculateBandwidth(nx, ny, elapsed);
    printResults(kernelInfo.name, bandwidth, theoretical_peak_bandwidth);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}

