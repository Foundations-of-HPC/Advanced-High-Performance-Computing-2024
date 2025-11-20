#include <stdio.h>
#include <cuda_runtime.h>
#include "helper.cuh"

int main() {
    float *h_M, *h_N, *h_P, *h_c_cpu; 
    float *d_M, *d_N, *d_P;
    
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    
    // Allocate host memory
    h_M = (float*)malloc(size);
    h_N = (float*)malloc(size);
    h_P = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size); // Memory for CPU result

    initMatrix(h_M, MATRIX_SIZE, MATRIX_SIZE, 1.0f);
    initMatrix(h_N, MATRIX_SIZE, MATRIX_SIZE, 2.0f);

    // CPU Matrix Multiplication
    double cpuTime = matrixMulCPU(h_M, h_N, h_c_cpu);
    
    // Allocate device memory
    cudaMalloc(&d_M, size);
    cudaMalloc(&d_N, size);
    cudaMalloc(&d_P, size);
    
    // Copy host matrices to device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    // Perform GPU matrix multiplication with different kernels
    double basicGpuTime = MatrixMultiplication(d_M, d_N, d_P, MATRIX_SIZE, 0);
    double unrolledGpuTime = MatrixMultiplication(d_M, d_N, d_P, MATRIX_SIZE, 1);
    
    // Calculate and print speedups
    printf("\nSpeedup Comparisons:\n");
    printf("Speedup (CPU / Basic GPU): %f\n", cpuTime / basicGpuTime);
    printf("Speedup (CPU / Unrolled GPU): %f\n", cpuTime / unrolledGpuTime);
    printf("Speedup (Basic GPU / Unrolled GPU): %f\n", basicGpuTime / unrolledGpuTime);
    
    // Copy result back to host (using the last kernel result)
    cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
    
    // Print a small portion of the result
    printf("\nPortion of the GPU result matrix:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_P[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }

    // Print a small portion of the CPU result matrix
    printf("\nPortion of the CPU result matrix:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", h_c_cpu[i * MATRIX_SIZE + j]);
        }
        printf("\n");
    }
    
    // Free memory
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_c_cpu);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    
    return 0;
}

