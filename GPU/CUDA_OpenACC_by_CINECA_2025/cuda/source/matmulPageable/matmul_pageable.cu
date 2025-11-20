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

// Function to calculate data transfer bandwidth
double calculate_transfer_bandwidth(long long total_bytes, double time_seconds) {
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
/*	{32, 32, 32},
	{64, 64, 64},
	{128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},*/
        {2048, 2048, 2048}
    };
	
	

    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("=== CUDA Matrix Multiplication with Pageable Memory Transfer ===\n");
    printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n", 
           "Size", "CPU_Time(ms)", "GPU_Time(ms)", "Transfer(ms)", "Speedup", "Bandwidth", "Validation", "GFLOPS");
    printf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-12s\n", 
           "----", "------------", "------------", "-----------", "-------", "---------", "----------", "------");
    
    for (int test = 0; test < num_sizes; test++) {
        int M = sizes[test][0];
        int N = sizes[test][1];
        int K = sizes[test][2];
        
        printf("\n=== Problem Size: %dx%dx%d ===\n", M, N, K);
        
        // Allocate host (CPU) memory - pageable
        float *h_A, *h_B, *h_C_gpu, *h_C_cpu;
        h_A = (float*)malloc(M * K * sizeof(float));
        h_B = (float*)malloc(K * N * sizeof(float));
        h_C_gpu = (float*)malloc(M * N * sizeof(float));
        h_C_cpu = (float*)malloc(M * N * sizeof(float));
        
        if (!h_A || !h_B || !h_C_gpu || !h_C_cpu) {
            printf("Failed to allocate host memory\n");
            exit(1);
        }
        
        // Allocate device (GPU) memory
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        
        // Initialize host matrices
        srand(42); // Fixed seed for reproducible results
        init_matrix(h_A, M * K);
        init_matrix(h_B, K * N);
        zero_matrix(h_C_gpu, M * N);
        zero_matrix(h_C_cpu, M * N);
        
        // CPU computation
        printf("Running CPU computation...\n");
        clock_t cpu_start = clock();
        naive_cpu_matmul(h_A, h_B, h_C_cpu, M, N, K);
        clock_t cpu_end = clock();
        double cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
        
        // GPU computation setup
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                      (M + blockSize.y - 1) / blockSize.y);
        
        printf("Grid size: (%d, %d), Block size: (%d, %d)\n", 
               gridSize.x, gridSize.y, blockSize.x, blockSize.y);
        
        // Create CUDA events for timing data transfers and computation
        cudaEvent_t transfer_start, transfer_stop, compute_start, compute_stop;
        CUDA_CHECK(cudaEventCreate(&transfer_start));
        CUDA_CHECK(cudaEventCreate(&transfer_stop));
        CUDA_CHECK(cudaEventCreate(&compute_start));
        CUDA_CHECK(cudaEventCreate(&compute_stop));
        
        // Time data transfer: Host to Device
        printf("Transferring data from Host to Device...\n");
        CUDA_CHECK(cudaEventRecord(transfer_start));
        
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C_gpu, M * N * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(transfer_stop));
        CUDA_CHECK(cudaEventSynchronize(transfer_stop));
        
        float h2d_time;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_time, transfer_start, transfer_stop));
        
        // Warm up GPU
        matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Reset device result matrix
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
        
        // Time GPU computation
        printf("Running GPU computation...\n");
        CUDA_CHECK(cudaEventRecord(compute_start));
        matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(compute_stop));
        CUDA_CHECK(cudaEventSynchronize(compute_stop));
        
        float gpu_compute_time;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_compute_time, compute_start, compute_stop));
        
        // Time data transfer: Device to Host
        printf("Transferring results from Device to Host...\n");
        CUDA_CHECK(cudaEventRecord(transfer_start));
        
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaEventRecord(transfer_stop));
        CUDA_CHECK(cudaEventSynchronize(transfer_stop));
        
        float d2h_time;
        CUDA_CHECK(cudaEventElapsedTime(&d2h_time, transfer_start, transfer_stop));
        
        // Calculate total times and metrics
        float total_transfer_time = h2d_time + d2h_time;
        float total_gpu_time = gpu_compute_time + total_transfer_time;
        
        double speedup = cpu_time / total_gpu_time;
        double compute_bandwidth = calculate_bandwidth(M, N, K, gpu_compute_time / 1000.0);
        
        // Calculate data transfer bandwidth
        long long h2d_bytes = (M * K + K * N + M * N) * sizeof(float);
        long long d2h_bytes = M * N * sizeof(float);
        double h2d_bandwidth = calculate_transfer_bandwidth(h2d_bytes, h2d_time / 1000.0);
        double d2h_bandwidth = calculate_transfer_bandwidth(d2h_bytes, d2h_time / 1000.0);
        
        // Calculate GFLOPS (2*M*N*K operations for matrix multiplication)
        double gflops = (2.0 * M * N * K) / (gpu_compute_time * 1e6);
        
        // Validate results
        printf("Validating results...\n");
        bool is_valid = validate_results(h_C_cpu, h_C_gpu, M * N);
        
        // Print detailed results
        printf("Problem Size: %dx%dx%d\n", M, N, K);
        printf("CPU Time: %.2f ms\n", cpu_time);
        printf("GPU Compute Time: %.2f ms\n", gpu_compute_time);
        printf("Host->Device Transfer: %.2f ms (%.2f GB/s)\n", h2d_time, h2d_bandwidth);
        printf("Device->Host Transfer: %.2f ms (%.2f GB/s)\n", d2h_time, d2h_bandwidth);
        printf("Total Transfer Time: %.2f ms\n", total_transfer_time);
        printf("Total GPU Time (Compute + Transfer): %.2f ms\n", total_gpu_time);
        printf("Speedup: %.2fx\n", speedup);
        printf("Compute Memory Bandwidth: %.2f GB/s\n", compute_bandwidth);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Validation: %s\n", is_valid ? "PASSED" : "FAILED");
        
        // Summary table row
        printf("%-8s %-12.2f %-12.2f %-12.2f %-12.2fx %-12.2f %-12s %-12.2f\n", 
               "Summary", cpu_time, total_gpu_time, total_transfer_time, speedup, 
               compute_bandwidth, is_valid ? "PASSED" : "FAILED", gflops);
        
        // Clean up events
        CUDA_CHECK(cudaEventDestroy(transfer_start));
        CUDA_CHECK(cudaEventDestroy(transfer_stop));
        CUDA_CHECK(cudaEventDestroy(compute_start));
        CUDA_CHECK(cudaEventDestroy(compute_stop));
        
        // Free device memory
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        
        // Free host memory
        free(h_A);
        free(h_B);
        free(h_C_gpu);
        free(h_C_cpu);
        
        printf("\n");
    }
       
    return 0;
}
