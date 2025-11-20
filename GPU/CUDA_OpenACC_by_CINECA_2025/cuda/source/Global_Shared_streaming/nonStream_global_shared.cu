#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include <math.h>

#define TILE_SIZE 16
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

// CPU matrix multiplication for validation
void cpu_matrix_mult(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// CUDA kernel with shared memory - highly optimized
__global__ void gpu_matrix_mult_shared(float *A, float *B, float *C, int N) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Global indices
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Number of tiles needed
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop through tiles
    for (int tile = 0; tile < numTiles; tile++) {
        // Load A tile (coalesced reads)
        int aRow = row;
        int aCol = tile * TILE_SIZE + tx;
        if (aRow < N && aCol < N) {
            As[ty][tx] = A[aRow * N + aCol];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Load B tile (coalesced reads)
        int bRow = tile * TILE_SIZE + ty;
        int bCol = col;
        if (bRow < N && bCol < N) {
            Bs[ty][tx] = B[bRow * N + bCol];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Synchronize to make sure tiles are loaded
        __syncthreads();
        
        // Compute partial sum using shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Simple GPU kernel without shared memory
__global__ void gpu_matrix_mult_naive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Validation function
int validate_result(float *cpu_result, float *gpu_result, int N) {
    float tolerance = 1e-3f;
    int errors = 0;
    
    for (int i = 0; i < N * N; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > tolerance) {
            errors++;
            if (errors <= 10) { // Print first 10 errors
                printf("Error at index %d: CPU=%.6f, GPU=%.6f, diff=%.6f\n", 
                       i, cpu_result[i], gpu_result[i], fabs(cpu_result[i] - gpu_result[i]));
            }
        }
    }
    
    return errors;
}

// Initialize matrix with random values
void init_matrix(float *mat, int N) {
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < N * N; i++) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Range [-1, 1]
    }
}

// Get current time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== CUDA Matrix Multiplication with Non-Default Stream ===\n\n");
    
    // Create a custom CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    printf("Created custom CUDA stream\n");
    
    // Problem sizes to test
    int sizes[] = {2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        printf("\nProblem Size: %d x %d\n", N, N);
        printf("Matrix elements: %d\n", N * N);
        
        size_t bytes = N * N * sizeof(float);
        printf("Memory per matrix: %.2f MB\n", bytes / (1024.0 * 1024.0));
        printf("Total memory: %.2f MB\n", 3.0 * bytes / (1024.0 * 1024.0));
        
        // Allocate host memory (pinned for better transfer performance)
        float *h_A, *h_B, *h_C_cpu, *h_C_gpu_naive, *h_C_gpu_shared;
        CHECK_CUDA(cudaMallocHost(&h_A, bytes));
        CHECK_CUDA(cudaMallocHost(&h_B, bytes));
        CHECK_CUDA(cudaMallocHost(&h_C_cpu, bytes));
        CHECK_CUDA(cudaMallocHost(&h_C_gpu_naive, bytes));
        CHECK_CUDA(cudaMallocHost(&h_C_gpu_shared, bytes));
        
        // Initialize matrices
        init_matrix(h_A, N);
        init_matrix(h_B, N);
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CHECK_CUDA(cudaMalloc(&d_A, bytes));
        CHECK_CUDA(cudaMalloc(&d_B, bytes));
        CHECK_CUDA(cudaMalloc(&d_C, bytes));
        
        // Copy data to device using the custom stream (asynchronous)
        CHECK_CUDA(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream));
        
        // Synchronize stream to ensure data transfer is complete before CPU computation
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // Skip CPU for large matrices to save time
        double cpu_time = 0.0;
        if (N <= 1024) {
            printf("\nRunning CPU computation...\n");
            double cpu_start = get_time();
            cpu_matrix_mult(h_A, h_B, h_C_cpu, N);
            cpu_time = get_time() - cpu_start;
            printf("CPU time: %.3f seconds\n", cpu_time);
        } else {
            printf("\nSkipping CPU computation for large matrix (N=%d)\n", N);
        }
        
        // GPU computation - Multiple runs for better accuracy
        printf("\nRunning GPU computation (naive) with custom stream...\n");
        dim3 block_size(TILE_SIZE, TILE_SIZE);
        dim3 grid_size((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
        
        // Create events for timing (associated with the stream)
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        // Multiple runs for more accurate timing
        int num_runs = 5;
        float total_naive_time = 0.0f;
        
        for (int run = 0; run < num_runs; run++) {
            CHECK_CUDA(cudaEventRecord(start, stream));
            gpu_matrix_mult_naive<<<grid_size, block_size, 0, stream>>>(d_A, d_B, d_C, N);
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            
            float run_time;
            CHECK_CUDA(cudaEventElapsedTime(&run_time, start, stop));
            total_naive_time += run_time;
        }
        
        double gpu_naive_time = (total_naive_time / num_runs) / 1000.0;
        CHECK_CUDA(cudaMemcpyAsync(h_C_gpu_naive, d_C, bytes, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("GPU naive time: %.3f seconds (avg of %d runs)\n", gpu_naive_time, num_runs);
        
        // GPU computation - Shared Memory (multiple runs) with custom stream
        printf("\nRunning GPU computation (shared memory) with custom stream...\n");
        
        float total_shared_time = 0.0f;
        
        for (int run = 0; run < num_runs; run++) {
            CHECK_CUDA(cudaEventRecord(start, stream));
            gpu_matrix_mult_shared<<<grid_size, block_size, 0, stream>>>(d_A, d_B, d_C, N);
            CHECK_CUDA(cudaEventRecord(stop, stream));
            CHECK_CUDA(cudaEventSynchronize(stop));
            
            float run_time;
            CHECK_CUDA(cudaEventElapsedTime(&run_time, start, stop));
            total_shared_time += run_time;
        }
        
        double gpu_shared_time = (total_shared_time / num_runs) / 1000.0;
        
        CHECK_CUDA(cudaMemcpyAsync(h_C_gpu_shared, d_C, bytes, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        printf("GPU shared memory time: %.3f seconds (avg of %d runs)\n", gpu_shared_time, num_runs);
        
        // Performance Analysis
        printf("\n=== PERFORMANCE ANALYSIS ===\n");
        double flops = 2.0 * N * N * N; // 2N^3 operations
        printf("Total FLOPs: %.2e\n", flops);
        
        if (N <= 1024) {
            printf("\nCPU Performance: %.2f GFLOPS\n", flops / cpu_time / 1e9);
            printf("Speedup (CPU vs GPU Naive): %.2fx\n", cpu_time / gpu_naive_time);
            printf("Speedup (CPU vs GPU Shared): %.2fx\n", cpu_time / gpu_shared_time);
        }
        
        printf("GPU Naive Performance: %.2f GFLOPS\n", flops / gpu_naive_time / 1e9);
        printf("GPU Shared Performance: %.2f GFLOPS\n", flops / gpu_shared_time / 1e9);
        printf("Speedup (GPU Naive vs GPU Shared): %.2fx\n", gpu_naive_time / gpu_shared_time);
        
        if (gpu_shared_time < gpu_naive_time) {
            printf("✓ Shared memory is FASTER by %.1fx!\n", gpu_naive_time / gpu_shared_time);
        } else {
            printf("✗ Shared memory is slower by %.1fx\n", gpu_shared_time / gpu_naive_time);
        }
        
        // Memory Bandwidth Analysis
        double mem_ops = 3.0 * bytes; // Read A, B and Write C
        printf("\nMemory Bandwidth (GPU Naive): %.2f GB/s\n", mem_ops / gpu_naive_time / 1e9);
        printf("Memory Bandwidth (GPU Shared): %.2f GB/s\n", mem_ops / gpu_shared_time / 1e9);
        
        // Validation
        printf("\n=== VALIDATION ===\n");
        if (N <= 1024) {
            int errors_naive = validate_result(h_C_cpu, h_C_gpu_naive, N);
            int errors_shared = validate_result(h_C_cpu, h_C_gpu_shared, N);
            
            printf("GPU Naive validation: %s (%d errors out of %d elements)\n", 
                   errors_naive == 0 ? "PASSED" : "FAILED", errors_naive, N*N);
            printf("GPU Shared validation: %s (%d errors out of %d elements)\n", 
                   errors_shared == 0 ? "PASSED" : "FAILED", errors_shared, N*N);
            
            if (errors_naive == 0 && errors_shared == 0) {
                printf("✓ All results match CPU computation!\n");
            }
        } else {
            // For large matrices, just compare GPU results
            int errors = validate_result(h_C_gpu_naive, h_C_gpu_shared, N);
            printf("GPU Naive vs GPU Shared validation: %s (%d errors out of %d elements)\n", 
                   errors == 0 ? "PASSED" : "FAILED", errors, N*N);
            if (errors == 0) {
                printf("✓ GPU results match!\n");
            }
        }
        
        // Cleanup host memory (pinned)
        cudaFreeHost(h_A); 
        cudaFreeHost(h_B); 
        cudaFreeHost(h_C_cpu); 
        cudaFreeHost(h_C_gpu_naive); 
        cudaFreeHost(h_C_gpu_shared);
        
        // Cleanup device memory
        cudaFree(d_A); 
        cudaFree(d_B); 
        cudaFree(d_C);
        
        // Cleanup events
        cudaEventDestroy(start); 
        cudaEventDestroy(stop);
        
        if (s < num_sizes - 1) {
            printf("\n=================================================\n");
        } else {
            printf("\n");
        }
    }
    
    // Destroy the custom stream
    CHECK_CUDA(cudaStreamDestroy(stream));
    printf("Destroyed custom CUDA stream\n");
    
    // Print device info
    printf("\n=== DEVICE INFORMATION ===\n");
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Warp Size: %d\n", prop.warpSize);
    printf("Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
    printf("Async Engine Count: %d\n", prop.asyncEngineCount);
    
    return 0;
}
