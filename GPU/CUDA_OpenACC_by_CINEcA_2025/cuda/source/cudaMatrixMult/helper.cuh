#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

const int MATRIX_SIZE = 2048;  // Define a constant for matrix size
const int BLOCK_SIZE = 16;     // Define block size for shared memory kernel


// Function to measure time
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// CPU matrix multiplication
double matrixMulCPU(float* a, float* b, float* c) {
    double start = cpuSecond();
    for (int row = 0; row < MATRIX_SIZE; ++row) {
        for (int col = 0; col < MATRIX_SIZE; ++col) {
            float val = 0;
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                val += a[row * MATRIX_SIZE + k] * b[k * MATRIX_SIZE + col];
            }
            c[row * MATRIX_SIZE + col] = val;
        }
    }
    double end = cpuSecond();
    double cpuTime = end - start;
    printf("CPU Matrix Multiplication Time: %f seconds\n", cpuTime);
    return cpuTime;
}

// Error checking macro
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// Host function to initialize matrix
void initMatrix(float *A, int n, int m, float c) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i * m + j] = c;
        }
    }
}

// Basic kernel for matrix multiplication
__global__ void basicMatrixMultiplicationKernel(float* M, float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width && col < Width) {
        float sum = 0;
        for (int k = 0; k < Width; ++k) {
            sum += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = sum;
    }
}

// Unrolled matrix multiplication kernel
__global__ void unrolledMatrixMultiplicationKernel(float *A, float *B, float *C, int n, int m, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Row index of C
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Column index of C

    if (i < n && j < p) {
        float sum = 0; // Changed to float
        for (int k = 0; k < m - 3; k += 4) {
            sum += A[i * m + k] * B[k * p + j] +
                   A[i * m + k + 1] * B[(k + 1) * p + j] +
                   A[i * m + k + 2] * B[(k + 2) * p + j] +
                   A[i * m + k + 3] * B[(k + 3) * p + j];
        }
        // Handle remaining elements
        for (int k = (m / 4) * 4; k < m; k++) {
            sum += A[i * m + k] * B[k * p + j];
        }
        C[i * p + j] = sum;
    }
}


// Updated host function for matrix multiplication
double MatrixMultiplication(float* M, float* N, float* P, int Width, int kernelType) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((Width + blockSize.x - 1) / blockSize.x, (Width + blockSize.y - 1) / blockSize.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    switch(kernelType) {
        case 0:
            basicMatrixMultiplicationKernel<<<gridSize, blockSize>>>(M, N, P, Width);
            break;
        case 1:
            unrolledMatrixMultiplicationKernel<<<gridSize, blockSize>>>(M, N, P, Width, Width, Width);
            break;
        default:
            printf("Invalid kernel type. Using basic kernel.\n");
            basicMatrixMultiplicationKernel<<<gridSize, blockSize>>>(M, N, P, Width);
    }

    cudaEventRecord(stop);
    
    cudaCheckError();
    cudaDeviceSynchronize();
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double gpuTime = milliseconds / 1000.0;
    
    const char* kernelName;
    switch(kernelType) {
        case 0: kernelName = "Basic"; break;
        case 1: kernelName = "Unrolled"; break;
        default: kernelName = "Unknown";
    }
    printf("%s GPU Matrix Multiplication Time: %f seconds\n", kernelName, gpuTime);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gpuTime;
}

// Function to calculate and print speedup
void calculateSpeedup(double cpuTime, double gpuTime) {
    double speedup = cpuTime / gpuTime;
    printf("Speedup (CPU time / GPU time): %f\n", speedup);
}

