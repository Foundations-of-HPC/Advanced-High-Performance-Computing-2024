#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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

// Alternative kernel with row-major threading
__global__ void matmul_kernel_xRow(float *A, float *B, float *C, int M, int N, int K) {
    int row = threadIdx.x + blockIdx.x * blockDim.x; 
    int col = threadIdx.y + blockIdx.y * blockDim.y; 
    
    if (col >= N || row >= M) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
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

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

int main() {
    
// Print device information
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("\n=== Device Information ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", (float)prop.totalGlobalMem / (1024*1024*1024));
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
    printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("Peak Memory Bandwidth: %.2f GB/s\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    
    printf("\n--------------------------------------------------------\n");

    printf("\n=== Memory Prefetching Strategy ===\n");
    printf("1. Data prefetched to CPU before CPU computation\n");
    printf("2. Data prefetched to GPU before GPU computation\n");
    printf("3. Results prefetched back to CPU for validation\n");
    printf("4. Uses cudaMemPrefetchAsync for optimal data locality\n");
   
	// Problem sizes to test
    
        int sizes[][3] = {
	{32, 32, 32},
	{64, 64, 64},
	{128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048}
    };
    
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("=== CUDA Matrix Multiplication with Unified Memory + Prefetching ===\n");
    printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s\n", 
           "Size", "CPU_Time(ms)", "GPU_Time(ms)", "Speedup", "Bandwidth", "Validation", "GFLOPS");
    printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s\n", 
           "----", "------------", "------------", "-------", "---------", "----------", "------");
    
    for (int test = 0; test < num_sizes; test++) {
        int M = sizes[test][0];
        int N = sizes[test][1];
        int K = sizes[test][2];
        
        printf("\n=== Problem Size: %dx%dx%d ===\n", M, N, K);
        
        // Allocate unified memory
        float *A, *B, *C_gpu, *C_cpu;
        
        CUDA_CHECK(cudaMallocManaged(&A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&C_gpu, M * N * sizeof(float)));
        CUDA_CHECK(cudaMallocManaged(&C_cpu, M * N * sizeof(float)));
        
        // Initialize matrices
        srand(42); // Fixed seed for reproducible results
        init_matrix(A, M * K);
        init_matrix(B, K * N);
        zero_matrix(C_gpu, M * N);
        zero_matrix(C_cpu, M * N);
        
        // Get device ID for prefetching
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        
        // Prefetch data to CPU for CPU computation
        printf("Prefetching data to CPU...\n");
        CUDA_CHECK(cudaMemPrefetchAsync(A, M * K * sizeof(float), cudaCpuDeviceId, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(B, K * N * sizeof(float), cudaCpuDeviceId, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(C_cpu, M * N * sizeof(float), cudaCpuDeviceId, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // CPU computation
        printf("Running CPU computation...\n");
        clock_t cpu_start = clock();
        naive_cpu_matmul(A, B, C_cpu, M, N, K);
        clock_t cpu_end = clock();
        double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
        
        // GPU computation setup
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                      (M + blockSize.y - 1) / blockSize.y);
        
        printf("Grid size: (%d, %d), Block size: (%d, %d)\n", 
               gridSize.x, gridSize.y, blockSize.x, blockSize.y);
        
        // Prefetch data to GPU for GPU computation
        printf("Prefetching data to GPU...\n");
        CUDA_CHECK(cudaMemPrefetchAsync(A, M * K * sizeof(float), device, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(B, K * N * sizeof(float), device, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(C_gpu, M * N * sizeof(float), device, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Warm up GPU
        matmul_kernel<<<gridSize, blockSize>>>(A, B, C_gpu, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Reset C_gpu for actual timing
        zero_matrix(C_gpu, M * N);
        
        // Prefetch again after reset to ensure data is on GPU
        CUDA_CHECK(cudaMemPrefetchAsync(C_gpu, M * N * sizeof(float), device, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // GPU computation with timing
        printf("Running GPU computation...\n");
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        matmul_kernel<<<gridSize, blockSize>>>(A, B, C_gpu, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float gpu_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_time_ms, start, stop));
        
        // Calculate metrics
        double speedup = cpu_time / gpu_time_ms;
        double bandwidth = calculate_bandwidth(M, N, K, gpu_time_ms / 1000.0);
        
        // Calculate GFLOPS (2*M*N*K operations for matrix multiplication)
        double gflops = (2.0 * M * N * K) / (gpu_time_ms * 1e6);
        
        // Validate results
        // Prefetch results back to CPU for validation
        printf("Prefetching results to CPU for validation...\n");
        CUDA_CHECK(cudaMemPrefetchAsync(C_gpu, M * N * sizeof(float), cudaCpuDeviceId, 0));
        CUDA_CHECK(cudaMemPrefetchAsync(C_cpu, M * N * sizeof(float), cudaCpuDeviceId, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        
        bool is_valid = validate_results(C_cpu, C_gpu, M * N);
        
        // Print results
        printf("Problem Size: %dx%dx%d\n", M, N, K);
        printf("CPU Time: %.2f ms\n", cpu_time);
        printf("GPU Time: %.2f ms\n", gpu_time_ms);
        printf("Speedup: %.2fx\n", speedup);
        printf("Memory Bandwidth: %.2f GB/s\n", bandwidth);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Validation: %s\n", is_valid ? "PASSED" : "FAILED");
        
        // Summary table row
        printf("%-8s %-12.2f %-12.2f %-12.2fx %-12.2f %-12s %-12.2f\n", 
               "Summary", cpu_time, gpu_time_ms, speedup, bandwidth, 
               is_valid ? "PASSED" : "FAILED", gflops);
        
        // Clean up events
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        // Free unified memory
        CUDA_CHECK(cudaFree(A));
        CUDA_CHECK(cudaFree(B));
        CUDA_CHECK(cudaFree(C_gpu));
        CUDA_CHECK(cudaFree(C_cpu));
        
        printf("\n");
    }
    
    
    return 0;
}
