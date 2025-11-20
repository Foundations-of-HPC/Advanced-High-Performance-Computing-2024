#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Configuration structure
typedef struct {
    int M, N, K;
    int block_size;
} MatrixConfig;

// Global configuration

//const MatrixConfig config = {1024, 1024, 1024, 16};
const MatrixConfig config = {2048, 2048, 2048, 16};
//const MatrixConfig config = {4096, 4096, 4096, 16};

// Structure to hold test data
typedef struct {
    float *h_A, *h_B, *h_C_cpu;
    double cpu_time;
    int size_A, size_B, size_C;
} TestData;

// Function declarations
void naive_cpu_matmul(float *A, float *B, float *C, int M, int N, int K);
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K);
__global__ void matmul_shared_kernel(float *A, float *B, float *C, int M, int N, int K);
void init_matrix(float *mat, int size);
void zero_matrix(float *mat, int size);
bool validate_results(float *cpu_result, float *gpu_result, int size, float tolerance);
double calculate_bandwidth(int M, int N, int K, double time_seconds);
double calculate_transfer_bandwidth(long long total_bytes, double time_seconds);

// Test initialization and CPU baseline computation
TestData* initialize_test_data(const MatrixConfig *cfg);
void cleanup_test_data(TestData *data);

// Memory management function declarations
int run_unified_memory(const MatrixConfig *cfg, TestData *data);
int run_pageable_memory(const MatrixConfig *cfg, TestData *data);
int run_pinned_memory(const MatrixConfig *cfg, TestData *data);
int run_shared_memory(const MatrixConfig *cfg, TestData *data);
int run_prefetch(const MatrixConfig *cfg, TestData *data);

// CPU naive matrix multiplication implementation
void naive_cpu_matmul(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Basic CUDA kernel for matrix multiplication
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    // Calculate the row and column for this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if the thread is within valid range
    if (col >= N || row >= M) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Shared memory optimized kernel
__global__ void matmul_shared_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    __shared__ float As[16][16]; // Using literal instead of TILE_SIZE for shared memory declaration
    __shared__ float Bs[16][16];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (col < N && tile * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to initialize matrix with random values
void init_matrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Random values between -1 and 1
    }
}

// Function to zero initialize matrix
void zero_matrix(float *mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = 0.0f;
    }
}

// Function to validate results by comparing CPU and GPU outputs
bool validate_results(float *cpu_result, float *gpu_result, int size, float tolerance = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            printf("Validation failed at index %d: CPU = %f, GPU = %f, diff = %f\n", 
                   i, cpu_result[i], gpu_result[i], fabs(cpu_result[i] - gpu_result[i]));
            return false;
        }
    }
    return true;
}

// Function to calculate memory bandwidth
double calculate_bandwidth(int M, int N, int K, double time_seconds) {
    // Total memory operations: read A (M*K), read B (K*N), write C (M*N)
    // Each operation is 4 bytes (float)
    long long total_bytes = (long long)(M * K + K * N + M * N) * sizeof(float);
    return (double)total_bytes / (time_seconds * 1e9); // GB/s
}

// Function to calculate data transfer bandwidth
double calculate_transfer_bandwidth(long long total_bytes, double time_seconds) {
    return (double)total_bytes / (time_seconds * 1e9); // GB/s
}

// Initialize test data and compute CPU baseline
TestData* initialize_test_data(const MatrixConfig *cfg) {
    TestData *data = (TestData*)malloc(sizeof(TestData));
    
    data->size_A = cfg->M * cfg->K * sizeof(float);
    data->size_B = cfg->K * cfg->N * sizeof(float);
    data->size_C = cfg->M * cfg->N * sizeof(float);
    
    // Allocate host memory for reference data
    data->h_A = (float*)malloc(data->size_A);
    data->h_B = (float*)malloc(data->size_B);
    data->h_C_cpu = (float*)malloc(data->size_C);
    
    // Initialize matrices with fixed seed for consistency
    srand(42); // Fixed seed for reproducible results
    init_matrix(data->h_A, cfg->M * cfg->K);
    init_matrix(data->h_B, cfg->K * cfg->N);
    zero_matrix(data->h_C_cpu, cfg->M * cfg->N);
    
    // Compute CPU baseline once
    printf("Computing CPU baseline...\n");
    clock_t cpu_start = clock();
    naive_cpu_matmul(data->h_A, data->h_B, data->h_C_cpu, cfg->M, cfg->N, cfg->K);
    clock_t cpu_end = clock();
    data->cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    
    printf("CPU baseline time: %.2f ms\n", data->cpu_time);
    
    return data;
}

// Cleanup test data
void cleanup_test_data(TestData *data) {
    if (data) {
        free(data->h_A);
        free(data->h_B);
        free(data->h_C_cpu);
        free(data);
    }
}

// 1. Unified Memory Implementation
int run_unified_memory(const MatrixConfig *cfg, TestData *test_data) {
    printf("\n=== Running Unified Memory Test ===\n");
    
    float *A, *B, *C;
    
    // Allocate unified memory
    CUDA_CHECK(cudaMallocManaged(&A, test_data->size_A));
    CUDA_CHECK(cudaMallocManaged(&B, test_data->size_B));
    CUDA_CHECK(cudaMallocManaged(&C, test_data->size_C));
    
    // Copy reference data to unified memory
    memcpy(A, test_data->h_A, test_data->size_A);
    memcpy(B, test_data->h_B, test_data->size_B);
    zero_matrix(C, cfg->M * cfg->N);
    
    // Setup kernel launch parameters
    dim3 blockSize(cfg->block_size, cfg->block_size);
    dim3 gridSize((cfg->N + blockSize.x - 1) / blockSize.x, 
                  (cfg->M + blockSize.y - 1) / blockSize.y);
    
    // Start timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch kernel
    matmul_kernel<<<gridSize, blockSize>>>(A, B, C, cfg->M, cfg->N, cfg->K);
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // Validate results
    bool is_correct = validate_results(test_data->h_C_cpu, C, cfg->M * cfg->N, 1e-2);
    
    // Calculate bandwidth
    double bandwidth = calculate_bandwidth(cfg->M, cfg->N, cfg->K, gpu_time / 1000.0);
    
    printf("GPU Time: %.2f ms\n", gpu_time);
    printf("CPU Time: %.2f ms\n", test_data->cpu_time);
    printf("Speedup: %.2fx\n", test_data->cpu_time / gpu_time);
    printf("Memory Bandwidth: %.2f GB/s\n", bandwidth);
    printf("Validation: %s\n", is_correct ? "PASSED" : "FAILED");
    
    // Cleanup
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return is_correct ? 0 : 1;
}

// 2. Pageable Memory Implementation
int run_pageable_memory(const MatrixConfig *cfg, TestData *test_data) {
    printf("\n=== Running Pageable Memory Test ===\n");
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    // Allocate host memory (pageable)
    h_A = (float*)malloc(test_data->size_A);
    h_B = (float*)malloc(test_data->size_B);
    h_C = (float*)malloc(test_data->size_C);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, test_data->size_A));
    CUDA_CHECK(cudaMalloc(&d_B, test_data->size_B));
    CUDA_CHECK(cudaMalloc(&d_C, test_data->size_C));
    
    // Copy reference data
    memcpy(h_A, test_data->h_A, test_data->size_A);
    memcpy(h_B, test_data->h_B, test_data->size_B);
    zero_matrix(h_C, cfg->M * cfg->N);
    
    // Setup kernel launch parameters
    dim3 blockSize(cfg->block_size, cfg->block_size);
    dim3 gridSize((cfg->N + blockSize.x - 1) / blockSize.x, 
                  (cfg->M + blockSize.y - 1) / blockSize.y);
    
    // Start timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, test_data->size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, test_data->size_B, cudaMemcpyHostToDevice));
    
    // Launch kernel
    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, cfg->M, cfg->N, cfg->K);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, test_data->size_C, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // Validate results
    bool is_correct = validate_results(test_data->h_C_cpu, h_C, cfg->M * cfg->N, 1e-2);
    
    // Calculate bandwidth
    long long total_bytes = test_data->size_A + test_data->size_B + test_data->size_C * 2; // Upload A, B and download C
    double transfer_bandwidth = calculate_transfer_bandwidth(total_bytes, gpu_time / 1000.0);
    double compute_bandwidth = calculate_bandwidth(cfg->M, cfg->N, cfg->K, gpu_time / 1000.0);
    
    printf("GPU Time (including transfers): %.2f ms\n", gpu_time);
    printf("CPU Time: %.2f ms\n", test_data->cpu_time);
    printf("Speedup: %.2fx\n", test_data->cpu_time / gpu_time);
    printf("Transfer Bandwidth: %.2f GB/s\n", transfer_bandwidth);
    printf("Compute Bandwidth: %.2f GB/s\n", compute_bandwidth);
    printf("Validation: %s\n", is_correct ? "PASSED" : "FAILED");
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return is_correct ? 0 : 1;
}

// 3. Pinned Memory Implementation
int run_pinned_memory(const MatrixConfig *cfg, TestData *test_data) {
    printf("\n=== Running Pinned Memory Test ===\n");
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    // Allocate pinned host memory
    CUDA_CHECK(cudaMallocHost(&h_A, test_data->size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, test_data->size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, test_data->size_C));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, test_data->size_A));
    CUDA_CHECK(cudaMalloc(&d_B, test_data->size_B));
    CUDA_CHECK(cudaMalloc(&d_C, test_data->size_C));
    
    // Copy reference data
    memcpy(h_A, test_data->h_A, test_data->size_A);
    memcpy(h_B, test_data->h_B, test_data->size_B);
    zero_matrix(h_C, cfg->M * cfg->N);
    
    // Setup kernel launch parameters
    dim3 blockSize(cfg->block_size, cfg->block_size);
    dim3 gridSize((cfg->N + blockSize.x - 1) / blockSize.x, 
                  (cfg->M + blockSize.y - 1) / blockSize.y);
    
    // Start timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, test_data->size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, test_data->size_B, cudaMemcpyHostToDevice));
    
    // Launch kernel
    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, cfg->M, cfg->N, cfg->K);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, test_data->size_C, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // Validate results
    bool is_correct = validate_results(test_data->h_C_cpu, h_C, cfg->M * cfg->N, 1e-2);
    
    // Calculate bandwidth
    long long total_bytes = test_data->size_A + test_data->size_B + test_data->size_C * 2; // Upload A, B and download C
    double transfer_bandwidth = calculate_transfer_bandwidth(total_bytes, gpu_time / 1000.0);
    double compute_bandwidth = calculate_bandwidth(cfg->M, cfg->N, cfg->K, gpu_time / 1000.0);
    
    printf("GPU Time (including transfers): %.2f ms\n", gpu_time);
    printf("CPU Time: %.2f ms\n", test_data->cpu_time);
    printf("Speedup: %.2fx\n", test_data->cpu_time / gpu_time);
    printf("Transfer Bandwidth: %.2f GB/s\n", transfer_bandwidth);
    printf("Compute Bandwidth: %.2f GB/s\n", compute_bandwidth);
    printf("Validation: %s\n", is_correct ? "PASSED" : "FAILED");
    
    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return is_correct ? 0 : 1;
}

// 4. Shared Memory Implementation
int run_shared_memory(const MatrixConfig *cfg, TestData *test_data) {
    printf("\n=== Running Shared Memory Test ===\n");
    
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    // Allocate pinned host memory for better performance
    CUDA_CHECK(cudaMallocHost(&h_A, test_data->size_A));
    CUDA_CHECK(cudaMallocHost(&h_B, test_data->size_B));
    CUDA_CHECK(cudaMallocHost(&h_C, test_data->size_C));
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, test_data->size_A));
    CUDA_CHECK(cudaMalloc(&d_B, test_data->size_B));
    CUDA_CHECK(cudaMalloc(&d_C, test_data->size_C));
    
    // Copy reference data
    memcpy(h_A, test_data->h_A, test_data->size_A);
    memcpy(h_B, test_data->h_B, test_data->size_B);
    zero_matrix(h_C, cfg->M * cfg->N);
    
    // Setup kernel launch parameters for tiled approach
    const int TILE_SIZE = 16;
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((cfg->N + TILE_SIZE - 1) / TILE_SIZE, 
                  (cfg->M + TILE_SIZE - 1) / TILE_SIZE);
    
    // Start timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, test_data->size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, test_data->size_B, cudaMemcpyHostToDevice));
    
    // Launch kernel with shared memory optimization
    matmul_shared_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, cfg->M, cfg->N, cfg->K);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, test_data->size_C, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // Validate results
    bool is_correct = validate_results(test_data->h_C_cpu, h_C, cfg->M * cfg->N, 1e-2);
    
    // Calculate bandwidth
    double compute_bandwidth = calculate_bandwidth(cfg->M, cfg->N, cfg->K, gpu_time / 1000.0);
    
    printf("GPU Time (with shared memory): %.2f ms\n", gpu_time);
    printf("CPU Time: %.2f ms\n", test_data->cpu_time);
    printf("Speedup: %.2fx\n", test_data->cpu_time / gpu_time);
    printf("Compute Bandwidth: %.2f GB/s\n", compute_bandwidth);
    printf("Validation: %s\n", is_correct ? "PASSED" : "FAILED");
    
    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return is_correct ? 0 : 1;
}

// 5. Prefetch Implementation
int run_prefetch(const MatrixConfig *cfg, TestData *test_data) {
    printf("\n=== Running Prefetch Test ===\n");
    
    // Check if device supports concurrent managed access
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    if (!prop.concurrentManagedAccess) {
        printf("Device does not support concurrent managed access. Skipping prefetch test.\n");
        return 0;
    }
    
    float *A, *B, *C;
    
    // Allocate unified memory
    CUDA_CHECK(cudaMallocManaged(&A, test_data->size_A));
    CUDA_CHECK(cudaMallocManaged(&B, test_data->size_B));
    CUDA_CHECK(cudaMallocManaged(&C, test_data->size_C));
    
    // Copy reference data
    memcpy(A, test_data->h_A, test_data->size_A);
    memcpy(B, test_data->h_B, test_data->size_B);
    zero_matrix(C, cfg->M * cfg->N);
    
    // Prefetch data to GPU
    CUDA_CHECK(cudaMemPrefetchAsync(A, test_data->size_A, device));
    CUDA_CHECK(cudaMemPrefetchAsync(B, test_data->size_B, device));
    CUDA_CHECK(cudaMemPrefetchAsync(C, test_data->size_C, device));
    
    // Setup kernel launch parameters
    dim3 blockSize(cfg->block_size, cfg->block_size);
    dim3 gridSize((cfg->N + blockSize.x - 1) / blockSize.x, 
                  (cfg->M + blockSize.y - 1) / blockSize.y);
    
    // Start timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    // Launch kernel
    matmul_kernel<<<gridSize, blockSize>>>(A, B, C, cfg->M, cfg->N, cfg->K);
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Prefetch result back to CPU for validation
    CUDA_CHECK(cudaMemPrefetchAsync(C, test_data->size_C, cudaCpuDeviceId));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float gpu_time;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_time, start, stop));
    
    // Validate results
    bool is_correct = validate_results(test_data->h_C_cpu, C, cfg->M * cfg->N, 1e-2);
    
    // Calculate bandwidth
    double bandwidth = calculate_bandwidth(cfg->M, cfg->N, cfg->K, gpu_time / 1000.0);
    
    printf("GPU Time (with prefetch): %.2f ms\n", gpu_time);
    printf("CPU Time: %.2f ms\n", test_data->cpu_time);
    printf("Speedup: %.2fx\n", test_data->cpu_time / gpu_time);
    printf("Memory Bandwidth: %.2f GB/s\n", bandwidth);
    printf("Validation: %s\n", is_correct ? "PASSED" : "FAILED");
    
    // Cleanup
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return is_correct ? 0 : 1;
}

// Main function to run all tests
int main() {
    printf("CUDA Matrix Multiplication Performance Comparison\n");
    printf("Matrix Size: %d x %d x %d\n", config.M, config.N, config.K);
    printf("Block Size: %d x %d\n", config.block_size, config.block_size);
    
    // Initialize test data and compute CPU baseline once
    TestData *test_data = initialize_test_data(&config);
    
    // Run all memory management tests
    int results[5];
    
    results[0] = run_unified_memory(&config, test_data);
    results[1] = run_pageable_memory(&config, test_data);
    results[2] = run_pinned_memory(&config, test_data);
    results[3] = run_shared_memory(&config, test_data);
    results[4] = run_prefetch(&config, test_data);
  
   // Summary
    printf("\n=== SUMMARY ===\n");
    printf("Unified Memory: %s\n", results[0] == 0 ? "PASSED" : "FAILED");
    printf("Pageable Memory: %s\n", results[1] == 0 ? "PASSED" : "FAILED");
    printf("Pinned Memory: %s\n", results[2] == 0 ? "PASSED" : "FAILED");
    printf("Shared Memory: %s\n", results[3] == 0 ? "PASSED" : "FAILED");
    printf("Prefetch: %s\n", results[4] == 0 ? "PASSED" : "FAILED");
    
    return 0;
}

