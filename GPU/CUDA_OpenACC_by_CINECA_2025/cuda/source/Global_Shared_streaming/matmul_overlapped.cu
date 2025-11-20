// Enhanced matmul with stream overlapping for better performance
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <sys/time.h>
#include <nvtx3/nvToolsExt.h>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

#define TILE_SIZE 16
#define NUM_STREAMS 4  // Multiple streams for better overlapping

void naive_cpu_matmul(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

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

__global__ void matmul_shared_kernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) {
        if (row < M && (k + tx) < K) {
            s_A[ty][tx] = A[row * K + k + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        if ((k + ty) < K && col < N) {
            s_B[ty][tx] = B[(k + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += s_A[ty][i] * s_B[i][tx];
        }

        __syncthreads();
    }

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

// Enhanced version with stream overlapping
void run_overlapped_matmul(float *A, float *B, float *C_result, int M, int N, int K, 
                          bool use_shared_memory = false) {
    
    // Create multiple streams for overlapping
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    // Create events for timing and synchronization
    cudaEvent_t start_event, stop_event;
    cudaEvent_t transfer_events[NUM_STREAMS * 2]; // For H2D and D2H events
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    
    for (int i = 0; i < NUM_STREAMS * 2; i++) {
        CHECK_CUDA(cudaEventCreate(&transfer_events[i]));
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // Configure grid and block dimensions
    dim3 block(use_shared_memory ? TILE_SIZE : 16, use_shared_memory ? TILE_SIZE : 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    printf("Using %s kernel with %d streams\n", 
           use_shared_memory ? "shared memory" : "global memory", NUM_STREAMS);
    
    nvtxRangePushA("Overlapped GPU Execution");
    CHECK_CUDA(cudaEventRecord(start_event, streams[0]));
    
    // Phase 1: Asynchronous memory transfers
    nvtxRangePushA("Async Memory Transfers");
    CHECK_CUDA(cudaEventRecord(transfer_events[0], streams[0]));
    CHECK_CUDA(cudaMemcpyAsync(d_A, A, M * K * sizeof(float), 
                              cudaMemcpyHostToDevice, streams[0]));
    CHECK_CUDA(cudaEventRecord(transfer_events[1], streams[0]));
    
    CHECK_CUDA(cudaMemcpyAsync(d_B, B, K * N * sizeof(float), 
                              cudaMemcpyHostToDevice, streams[1]));
    CHECK_CUDA(cudaEventRecord(transfer_events[2], streams[1]));
    
    CHECK_CUDA(cudaMemsetAsync(d_C, 0, M * N * sizeof(float), streams[2]));
    CHECK_CUDA(cudaEventRecord(transfer_events[3], streams[2]));
    nvtxRangePop();
    
    // Phase 2: Wait for transfers to complete, then launch kernel
    nvtxRangePushA("Kernel Execution");
    CHECK_CUDA(cudaStreamWaitEvent(streams[3], transfer_events[1], 0)); // Wait for A transfer
    CHECK_CUDA(cudaStreamWaitEvent(streams[3], transfer_events[2], 0)); // Wait for B transfer
    CHECK_CUDA(cudaStreamWaitEvent(streams[3], transfer_events[3], 0)); // Wait for C memset
    
    // Launch the appropriate kernel
    if (use_shared_memory) {
        matmul_shared_kernel<<<grid, block, 0, streams[3]>>>(d_A, d_B, d_C, M, N, K);
    } else {
        matmul_kernel<<<grid, block, 0, streams[3]>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(transfer_events[4], streams[3])); // Kernel completion
    nvtxRangePop();
    
    // Phase 3: Asynchronous result transfer
    nvtxRangePushA("Async Result Transfer");
    CHECK_CUDA(cudaStreamWaitEvent(streams[0], transfer_events[4], 0)); // Wait for kernel
    CHECK_CUDA(cudaMemcpyAsync(C_result, d_C, M * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, streams[0]));
    CHECK_CUDA(cudaEventRecord(transfer_events[5], streams[0]));
    nvtxRangePop();
    
    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }
    
    CHECK_CUDA(cudaEventRecord(stop_event, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop_event));
    nvtxRangePop();
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    
    for (int i = 0; i < NUM_STREAMS * 2; i++) {
        CHECK_CUDA(cudaEventDestroy(transfer_events[i]));
    }
    
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
}

// Double-buffered version for even better overlapping
void run_double_buffered_matmul(float *A, float *B, float *C_result, int M, int N, int K) {
    
    // Create two streams for ping-pong execution
    cudaStream_t stream_compute, stream_transfer;
    CHECK_CUDA(cudaStreamCreate(&stream_compute));
    CHECK_CUDA(cudaStreamCreate(&stream_transfer));
    
    // Create events for synchronization
    cudaEvent_t start_event, stop_event;
    cudaEvent_t compute_done, transfer_done;
    CHECK_CUDA(cudaEventCreate(&start_event));
    CHECK_CUDA(cudaEventCreate(&stop_event));
    CHECK_CUDA(cudaEventCreate(&compute_done));
    CHECK_CUDA(cudaEventCreate(&transfer_done));
    
    // Allocate double-buffered device memory
    float *d_A[2], *d_B[2], *d_C[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMalloc(&d_A[i], M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B[i], K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], M * N * sizeof(float)));
    }
    
    // Split work into chunks for demonstration (could be row-wise or block-wise)
    int chunk_size = M / 2;
    int remaining = M - chunk_size;
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid1((N + block.x - 1) / block.x, (chunk_size + block.y - 1) / block.y);
    dim3 grid2((N + block.x - 1) / block.x, (remaining + block.y - 1) / block.y);
    
    printf("Using double-buffered execution with chunk sizes: %d, %d\n", chunk_size, remaining);
    
    nvtxRangePushA("Double-Buffered Execution");
    CHECK_CUDA(cudaEventRecord(start_event, stream_compute));
    
    // First chunk: Transfer + Compute
    nvtxRangePushA("Chunk 1");
    CHECK_CUDA(cudaMemcpyAsync(d_A[0], A, chunk_size * K * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(d_B[0], B, K * N * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_transfer));
    CHECK_CUDA(cudaMemsetAsync(d_C[0], 0, chunk_size * N * sizeof(float), stream_transfer));
    CHECK_CUDA(cudaEventRecord(transfer_done, stream_transfer));
    
    CHECK_CUDA(cudaStreamWaitEvent(stream_compute, transfer_done, 0));
    matmul_shared_kernel<<<grid1, block, 0, stream_compute>>>(d_A[0], d_B[0], d_C[0], chunk_size, N, K);
    CHECK_CUDA(cudaEventRecord(compute_done, stream_compute));
    nvtxRangePop();
    
    // Second chunk: Overlap transfer of chunk 2 with computation
    nvtxRangePushA("Chunk 2 - Overlapped");
    CHECK_CUDA(cudaMemcpyAsync(d_A[1], A + chunk_size * K, remaining * K * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_transfer));
    CHECK_CUDA(cudaMemcpyAsync(d_B[1], B, K * N * sizeof(float), 
                              cudaMemcpyHostToDevice, stream_transfer));
    CHECK_CUDA(cudaMemsetAsync(d_C[1], 0, remaining * N * sizeof(float), stream_transfer));
    
    // Start transferring result of chunk 1 while chunk 2 is being prepared
    CHECK_CUDA(cudaStreamWaitEvent(stream_transfer, compute_done, 0));
    CHECK_CUDA(cudaMemcpyAsync(C_result, d_C[0], chunk_size * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, stream_transfer));
    
    // Compute chunk 2
    CHECK_CUDA(cudaStreamSynchronize(stream_transfer)); // Ensure chunk 2 data is ready
    matmul_shared_kernel<<<grid2, block, 0, stream_compute>>>(d_A[1], d_B[1], d_C[1], remaining, N, K);
    
    // Transfer result of chunk 2
    CHECK_CUDA(cudaMemcpyAsync(C_result + chunk_size * N, d_C[1], remaining * N * sizeof(float), 
                              cudaMemcpyDeviceToHost, stream_compute));
    nvtxRangePop();
    
    CHECK_CUDA(cudaStreamSynchronize(stream_compute));
    CHECK_CUDA(cudaStreamSynchronize(stream_transfer));
    
    CHECK_CUDA(cudaEventRecord(stop_event, stream_compute));
    CHECK_CUDA(cudaEventSynchronize(stop_event));
    nvtxRangePop();
    
    // Cleanup
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaFree(d_A[i]));
        CHECK_CUDA(cudaFree(d_B[i]));
        CHECK_CUDA(cudaFree(d_C[i]));
    }
    
    CHECK_CUDA(cudaStreamDestroy(stream_compute));
    CHECK_CUDA(cudaStreamDestroy(stream_transfer));
    CHECK_CUDA(cudaEventDestroy(start_event));
    CHECK_CUDA(cudaEventDestroy(stop_event));
    CHECK_CUDA(cudaEventDestroy(compute_done));
    CHECK_CUDA(cudaEventDestroy(transfer_done));
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
    printf("Enhanced version with stream overlapping...\n");

    // Allocate pinned host memory
    float *A, *B, *C_cpu, *C_gpu_overlapped, *C_gpu_shared_overlapped, *C_gpu_double_buffered;
    CHECK_CUDA(cudaMallocHost(&A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_cpu, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_gpu_overlapped, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_gpu_shared_overlapped, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_gpu_double_buffered, M * N * sizeof(float)));

    // Initialize matrices
    init_matrix(A, M * K, 1.0f);
    init_matrix(B, K * N, 1.0f);
    init_matrix(C_cpu, M * N, 0.0f);
    init_matrix(C_gpu_overlapped, M * N, 0.0f);
    init_matrix(C_gpu_shared_overlapped, M * N, 0.0f);
    init_matrix(C_gpu_double_buffered, M * N, 0.0f);

    // CPU computation for reference
    printf("\n--- CPU Computation ---\n");
    nvtxRangePushA("CPU Computation");
    double start_cpu = get_time_ms();
    naive_cpu_matmul(A, B, C_cpu, M, N, K);
    double end_cpu = get_time_ms();
    nvtxRangePop();
    printf("CPU execution time: %.2f ms\n", end_cpu - start_cpu);

    // GPU overlapped execution (global memory)
    printf("\n--- GPU Overlapped (Global Memory) ---\n");
    double start_overlapped = get_time_ms();
    run_overlapped_matmul(A, B, C_gpu_overlapped, M, N, K, false);
    double end_overlapped = get_time_ms();
    
    bool valid_overlapped = compare_matrices(C_cpu, C_gpu_overlapped, M * N);
    printf("Results: %s\n", valid_overlapped ? "PASS" : "FAIL");
    printf("Execution time: %.2f ms\n", end_overlapped - start_overlapped);
    printf("Speedup: %.2fx\n", (end_cpu - start_cpu) / (end_overlapped - start_overlapped));

    // GPU overlapped execution (shared memory)
    printf("\n--- GPU Overlapped (Shared Memory) ---\n");
    double start_shared_overlapped = get_time_ms();
    run_overlapped_matmul(A, B, C_gpu_shared_overlapped, M, N, K, true);
    double end_shared_overlapped = get_time_ms();
    
    bool valid_shared_overlapped = compare_matrices(C_cpu, C_gpu_shared_overlapped, M * N);
    printf("Results: %s\n", valid_shared_overlapped ? "PASS" : "FAIL");
    printf("Execution time: %.2f ms\n", end_shared_overlapped - start_shared_overlapped);
    printf("Speedup: %.2fx\n", (end_cpu - start_cpu) / (end_shared_overlapped - start_shared_overlapped));

    // Double-buffered execution
    printf("\n--- GPU Double-Buffered ---\n");
    double start_double_buffered = get_time_ms();
    run_double_buffered_matmul(A, B, C_gpu_double_buffered, M, N, K);
    double end_double_buffered = get_time_ms();
    
    bool valid_double_buffered = compare_matrices(C_cpu, C_gpu_double_buffered, M * N);
    printf("Results: %s\n", valid_double_buffered ? "PASS" : "FAIL");
    printf("Execution time: %.2f ms\n", end_double_buffered - start_double_buffered);
    printf("Speedup: %.2fx\n", (end_cpu - start_cpu) / (end_double_buffered - start_double_buffered));

    // Performance summary
    printf("\n=== Performance Summary ===\n");
    printf("CPU Time: %.2f ms\n", end_cpu - start_cpu);
    printf("GPU Overlapped (Global):  %.2f ms (%.2fx speedup)\n", 
           end_overlapped - start_overlapped,
           (end_cpu - start_cpu) / (end_overlapped - start_overlapped));
    printf("GPU Overlapped (Shared):  %.2f ms (%.2fx speedup)\n",
           end_shared_overlapped - start_shared_overlapped,
           (end_cpu - start_cpu) / (end_shared_overlapped - start_shared_overlapped));
    printf("GPU Double-Buffered:      %.2f ms (%.2fx speedup)\n",
           end_double_buffered - start_double_buffered,
           (end_cpu - start_cpu) / (end_double_buffered - start_double_buffered));

    // Memory bandwidth analysis
    double data_size_mb = (M * K + K * N + M * N) * sizeof(float) / (1024.0 * 1024.0);
    printf("\nMemory Transfer Analysis:\n");
    printf("Total data size: %.2f MB\n", data_size_mb);
    printf("Overlapped version effective bandwidth: %.2f GB/s\n",
           data_size_mb / (end_shared_overlapped - start_shared_overlapped));

    // Cleanup
    CHECK_CUDA(cudaFreeHost(A));
    CHECK_CUDA(cudaFreeHost(B));
    CHECK_CUDA(cudaFreeHost(C_cpu));
    CHECK_CUDA(cudaFreeHost(C_gpu_overlapped));
    CHECK_CUDA(cudaFreeHost(C_gpu_shared_overlapped));
    CHECK_CUDA(cudaFreeHost(C_gpu_double_buffered));

    return 0;
}
