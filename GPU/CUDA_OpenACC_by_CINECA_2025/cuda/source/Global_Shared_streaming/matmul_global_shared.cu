// Matmul_pinned.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <sys/time.h>
#include <nvtx3/nvToolsExt.h>  // For NVTX profiling

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

// Tile size for shared memory blocks
#define TILE_SIZE 16

void naive_cpu_matmul(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Basic global memory kernel
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < N && row < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Shared memory optimized kernel
__global__ void matmul_shared_kernel(float *A, float *B, float *C, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Output element position
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;

    // Loop over tiles in the K dimension
    for (int k = 0; k < K; k += TILE_SIZE) {
        // Load tile of A into shared memory
        if (row < M && (k + tx) < K) {
            s_A[ty][tx] = A[row * K + k + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        if ((k + ty) < K && col < N) {
            s_B[ty][tx] = B[(k + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product from tiles
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += s_A[ty][i] * s_B[i][tx];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void init_matrix(float *mat, int size, float val) {
    for (int i = 0; i < size; i++) {
        mat[i] = val;
    }
}

bool compare_matrices(float *ref, float *test, int size, float eps = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (fabs(ref[i] - test[i]) > eps) {
            printf("Mismatch at index %d: CPU=%.3f, GPU=%.3f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s M N K\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Using pinned memory for faster transfers...\n");

    // Allocate pinned (page-locked) host memory for faster transfers
    float *A, *B, *C_cpu, *C_gpu_global, *C_gpu_shared;
    CHECK_CUDA(cudaMallocHost(&A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_cpu, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_gpu_global, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_gpu_shared, M * N * sizeof(float)));

    // Initialize matrices
    init_matrix(A, M * K, 1.0f);
    init_matrix(B, K * N, 1.0f);
    init_matrix(C_cpu, M * N, 0.0f);
    init_matrix(C_gpu_global, M * N, 0.0f);
    init_matrix(C_gpu_shared, M * N, 0.0f);

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));

    // Create CUDA streams for potential asynchronous operations
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Configure kernels
    dim3 block_global(16, 16);
    dim3 grid_global((N + block_global.x - 1) / block_global.x, (M + block_global.y - 1) / block_global.y);
    
    dim3 block_shared(TILE_SIZE, TILE_SIZE);
    dim3 grid_shared((N + block_shared.x - 1) / block_shared.x, (M + block_shared.y - 1) / block_shared.y);

    // ================= CPU Computation =================
    printf("\n--- CPU Computation ---\n");
    nvtxRangePushA("CPU Computation");
    double start_cpu = get_time_ms();
    naive_cpu_matmul(A, B, C_cpu, M, N, K);
    double end_cpu = get_time_ms();
    nvtxRangePop();
    printf("CPU execution time: %.2f ms\n", end_cpu - start_cpu);

    // ================= GPU Global Memory Kernel =================
    printf("\n--- GPU Global Memory Kernel ---\n");
    nvtxRangePushA("GPU Global Memory Total");
    double start_gpu_global = get_time_ms();
    
    // Memory transfer: Host to Device
    nvtxRangePushA("H2D Transfer (Global)");
    CHECK_CUDA(cudaMemcpyAsync(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(d_C, 0, M * N * sizeof(float), stream));
    nvtxRangePop();
    
    // Kernel execution
    nvtxRangePushA("Global Memory Kernel");
    CHECK_CUDA(cudaEventRecord(start_event, stream));
    matmul_kernel<<<grid_global, block_global, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop_event, stream));
    nvtxRangePop();
    
    // Memory transfer: Device to Host
    nvtxRangePushA("D2H Transfer (Global)");
    CHECK_CUDA(cudaMemcpyAsync(C_gpu_global, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    nvtxRangePop();
    
    double end_gpu_global = get_time_ms();
    nvtxRangePop();

    // Get kernel time for global memory version
    float kernel_time_global;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time_global, start_event, stop_event));

    // Verify global memory results
    bool valid_global = compare_matrices(C_cpu, C_gpu_global, M * N);
    
    printf("Results: %s\n", valid_global ? "PASS" : "FAIL");
    printf("  Kernel time: %.2f ms\n", kernel_time_global);
    printf("  Total GPU time (incl. transfer): %.2f ms\n", end_gpu_global - start_gpu_global);
    printf("  Speedup (kernel): %.2fx\n", (end_cpu - start_cpu) / kernel_time_global);
    printf("  Speedup (total): %.2fx\n", (end_cpu - start_cpu) / (end_gpu_global - start_gpu_global));

    // ================= GPU Shared Memory Kernel =================
    printf("\n--- GPU Shared Memory Kernel ---\n");
    nvtxRangePushA("GPU Shared Memory Total");
    double start_gpu_shared = get_time_ms();
    
    // Memory transfer: Host to Device (reuse existing data)
    nvtxRangePushA("H2D Transfer (Shared)");
    CHECK_CUDA(cudaMemsetAsync(d_C, 0, M * N * sizeof(float), stream));
    nvtxRangePop();
    
    // Kernel execution with shared memory
    nvtxRangePushA("Shared Memory Kernel");
    CHECK_CUDA(cudaEventRecord(start_event, stream));
    matmul_shared_kernel<<<grid_shared, block_shared, 0, stream>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop_event, stream));
    nvtxRangePop();
    
    // Memory transfer: Device to Host
    nvtxRangePushA("D2H Transfer (Shared)");
    CHECK_CUDA(cudaMemcpyAsync(C_gpu_shared, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    nvtxRangePop();
    
    double end_gpu_shared = get_time_ms();
    nvtxRangePop();

    // Get kernel time for shared memory version
    float kernel_time_shared;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time_shared, start_event, stop_event));

    // Verify shared memory results
    bool valid_shared = compare_matrices(C_cpu, C_gpu_shared, M * N);
    
    printf("Tile size: %dx%d\n", TILE_SIZE, TILE_SIZE);
    printf("Results: %s\n", valid_shared ? "PASS" : "FAIL");
    printf("  Kernel time: %.2f ms\n", kernel_time_shared);
    printf("  Total GPU time (incl. transfer): %.2f ms\n", end_gpu_shared - start_gpu_shared);
    printf("  Speedup (kernel): %.2fx\n", (end_cpu - start_cpu) / kernel_time_shared);
    printf("  Speedup (total): %.2fx\n", (end_cpu - start_cpu) / (end_gpu_shared - start_gpu_shared));

    // ================= Performance Summary =================
    printf("\n=== Performance Summary ===\n");
    printf("CPU Time: %.2f ms\n", end_cpu - start_cpu);
    printf("GPU Global Memory:\n");
    printf("  - Kernel: %.2f ms (%.2fx speedup)\n", kernel_time_global, 
           (end_cpu - start_cpu) / kernel_time_global);
    printf("  - Total:  %.2f ms (%.2fx speedup)\n", end_gpu_global - start_gpu_global,
           (end_cpu - start_cpu) / (end_gpu_global - start_gpu_global));
    printf("GPU Shared Memory:\n");
    printf("  - Kernel: %.2f ms (%.2fx speedup)\n", kernel_time_shared,
           (end_cpu - start_cpu) / kernel_time_shared);
    printf("  - Total:  %.2f ms (%.2fx speedup)\n", end_gpu_shared - start_gpu_shared,
           (end_cpu - start_cpu) / (end_gpu_shared - start_gpu_shared));
    
    if (kernel_time_shared < kernel_time_global) {
        printf("Shared memory kernel is %.2fx faster than global memory kernel\n",
               kernel_time_global / kernel_time_shared);
    } else {
        printf("Global memory kernel is %.2fx faster than shared memory kernel\n",
               kernel_time_shared / kernel_time_global);
    }

    // Memory transfer analysis
    double data_size_mb = (M * K + K * N + M * N) * sizeof(float) / (1024.0 * 1024.0);
    double transfer_time_global = (end_gpu_global - start_gpu_global) - kernel_time_global;
    double transfer_time_shared = (end_gpu_shared - start_gpu_shared) - kernel_time_shared;
    
    printf("\nMemory Transfer Analysis:\n");
    printf("Total data transferred: %.2f MB\n", data_size_mb);
    printf("Global memory version transfer time: %.2f ms (%.2f GB/s)\n", 
           transfer_time_global, data_size_mb / transfer_time_global);
    printf("Shared memory version transfer time: %.2f ms (%.2f GB/s)\n",
           transfer_time_shared, data_size_mb / transfer_time_shared);

    // Cleanup pinned memory
    CHECK_CUDA(cudaFreeHost(A));
    CHECK_CUDA(cudaFreeHost(B));
    CHECK_CUDA(cudaFreeHost(C_cpu));
    CHECK_CUDA(cudaFreeHost(C_gpu_global));
    CHECK_CUDA(cudaFreeHost(C_gpu_shared));
    
    // Cleanup device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    // Cleanup CUDA objects
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}
