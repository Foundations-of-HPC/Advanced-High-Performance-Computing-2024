// True overlapped CUDA matrix multiplication with data/computation overlap
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
#define NUM_STREAMS 3  // One for compute, two for transfers

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

// True overlapped execution with chunked processing
void run_true_overlapped_matmul(float *A, float *B, float *C_result, int M, int N, int K, 
                                bool use_shared_memory = false) {
    
    // Calculate chunk size - should be large enough to hide memory transfer latency
    int num_chunks = 4;  // Adjust based on your GPU memory and requirements
    int chunk_rows = (M + num_chunks - 1) / num_chunks;
    
    printf("Processing with %d chunks of %d rows each\n", num_chunks, chunk_rows);
    
    // Create streams: one for compute, others for async transfers
    cudaStream_t compute_stream, h2d_stream, d2h_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&h2d_stream));
    CHECK_CUDA(cudaStreamCreate(&d2h_stream));
    
    // Create events for synchronization between streams
    cudaEvent_t *h2d_done, *compute_done;
    h2d_done = (cudaEvent_t*)malloc(num_chunks * sizeof(cudaEvent_t));
    compute_done = (cudaEvent_t*)malloc(num_chunks * sizeof(cudaEvent_t));
    
    for (int i = 0; i < num_chunks; i++) {
        CHECK_CUDA(cudaEventCreate(&h2d_done[i]));
        CHECK_CUDA(cudaEventCreate(&compute_done[i]));
    }
    
    // Allocate device memory for double buffering
    float *d_A[2], *d_B, *d_C[2];
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaMalloc(&d_A[i], chunk_rows * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], chunk_rows * N * sizeof(float)));
    }
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    
    // Transfer B matrix once (it's used by all chunks)
    nvtxRangePushA("Transfer B matrix");
    CHECK_CUDA(cudaMemcpyAsync(d_B, B, K * N * sizeof(float), 
                              cudaMemcpyHostToDevice, h2d_stream));
    CHECK_CUDA(cudaStreamSynchronize(h2d_stream));
    nvtxRangePop();
    
    // Configure grid and block dimensions
    dim3 block(use_shared_memory ? TILE_SIZE : 16, use_shared_memory ? TILE_SIZE : 16);
    
    nvtxRangePushA("Overlapped Chunked Execution");
    
    // Process chunks with overlapping
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int current_chunk_rows = min(chunk_rows, M - chunk * chunk_rows);
        if (current_chunk_rows <= 0) break;
        
        int buffer_idx = chunk % 2;  // Ping-pong between buffers
        
        int start_row = chunk * chunk_rows;
        float *chunk_A = A + start_row * K;
        float *chunk_C = C_result + start_row * N;
        
        dim3 grid((N + block.x - 1) / block.x, 
                  (current_chunk_rows + block.y - 1) / block.y);
        
        nvtxRangePushA("Chunk Processing");
        
        // Phase 1: Transfer current chunk A data (async)
        CHECK_CUDA(cudaMemcpyAsync(d_A[buffer_idx], chunk_A, 
                                  current_chunk_rows * K * sizeof(float), 
                                  cudaMemcpyHostToDevice, h2d_stream));
        CHECK_CUDA(cudaEventRecord(h2d_done[chunk], h2d_stream));
        
        // Phase 2: Wait for transfer, then compute (overlapped with next transfer)
        CHECK_CUDA(cudaStreamWaitEvent(compute_stream, h2d_done[chunk], 0));
        
        if (use_shared_memory) {
            matmul_shared_kernel<<<grid, block, 0, compute_stream>>>(
                d_A[buffer_idx], d_B, d_C[buffer_idx], current_chunk_rows, N, K);
        } else {
            matmul_kernel<<<grid, block, 0, compute_stream>>>(
                d_A[buffer_idx], d_B, d_C[buffer_idx], current_chunk_rows, N, K);
        }
        CHECK_CUDA(cudaEventRecord(compute_done[chunk], compute_stream));
        
        // Phase 3: Transfer result back (async, overlapped with next computation)
        CHECK_CUDA(cudaStreamWaitEvent(d2h_stream, compute_done[chunk], 0));
        CHECK_CUDA(cudaMemcpyAsync(chunk_C, d_C[buffer_idx], 
                                  current_chunk_rows * N * sizeof(float), 
                                  cudaMemcpyDeviceToHost, d2h_stream));
        
        nvtxRangePop();
    }
    
    // Synchronize all streams to ensure completion
    CHECK_CUDA(cudaStreamSynchronize(compute_stream));
    CHECK_CUDA(cudaStreamSynchronize(h2d_stream));
    CHECK_CUDA(cudaStreamSynchronize(d2h_stream));
    
    nvtxRangePop();
    
    // Cleanup
    for (int i = 0; i < 2; i++) {
        CHECK_CUDA(cudaFree(d_A[i]));
        CHECK_CUDA(cudaFree(d_C[i]));
    }
    CHECK_CUDA(cudaFree(d_B));
    
    for (int i = 0; i < num_chunks; i++) {
        CHECK_CUDA(cudaEventDestroy(h2d_done[i]));
        CHECK_CUDA(cudaEventDestroy(compute_done[i]));
    }
    free(h2d_done);
    free(compute_done);
    
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaStreamDestroy(h2d_stream));
    CHECK_CUDA(cudaStreamDestroy(d2h_stream));
}

// Pipelined version with explicit overlap demonstration
void run_pipelined_matmul(float *A, float *B, float *C_result, int M, int N, int K) {
    
    int num_chunks = 6;  // More chunks for better pipeline
    int chunk_rows = (M + num_chunks - 1) / num_chunks;
    
    printf("Pipelined execution with %d chunks of %d rows each\n", num_chunks, chunk_rows);
    
    // Create dedicated streams
    cudaStream_t stream_h2d, stream_compute, stream_d2h;
    CHECK_CUDA(cudaStreamCreate(&stream_h2d));
    CHECK_CUDA(cudaStreamCreate(&stream_compute));  
    CHECK_CUDA(cudaStreamCreate(&stream_d2h));
    
    // Events for pipeline synchronization
    cudaEvent_t *transfer_ready, *compute_ready;
    transfer_ready = (cudaEvent_t*)malloc(num_chunks * sizeof(cudaEvent_t));
    compute_ready = (cudaEvent_t*)malloc(num_chunks * sizeof(cudaEvent_t));
    
    for (int i = 0; i < num_chunks; i++) {
        CHECK_CUDA(cudaEventCreate(&transfer_ready[i]));
        CHECK_CUDA(cudaEventCreate(&compute_ready[i]));
    }
    
    // Triple buffering for smooth pipeline
    float *d_A[3], *d_B, *d_C[3];
    for (int i = 0; i < 3; i++) {
        CHECK_CUDA(cudaMalloc(&d_A[i], chunk_rows * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C[i], chunk_rows * N * sizeof(float)));
    }
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    
    // Pre-transfer B matrix
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    nvtxRangePushA("Pipelined Execution");
    
    // Pipeline: Transfer → Compute → Transfer Back
    // All three stages run concurrently on different chunks
    
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int current_chunk_rows = min(chunk_rows, M - chunk * chunk_rows);
        if (current_chunk_rows <= 0) break;
        
        int buffer_idx = chunk % 3;
        int start_row = chunk * chunk_rows;
        
        dim3 grid((N + block.x - 1) / block.x, 
                  (current_chunk_rows + block.y - 1) / block.y);
        
        // Stage 1: H2D Transfer (overlapped with previous compute and D2H)
        nvtxRangePushA("H2D Transfer");
        CHECK_CUDA(cudaMemcpyAsync(d_A[buffer_idx], A + start_row * K, 
                                  current_chunk_rows * K * sizeof(float), 
                                  cudaMemcpyHostToDevice, stream_h2d));
        CHECK_CUDA(cudaEventRecord(transfer_ready[chunk], stream_h2d));
        nvtxRangePop();
        
        // Stage 2: Compute (wait for current transfer, overlapped with next transfer)
        nvtxRangePushA("Compute");
        CHECK_CUDA(cudaStreamWaitEvent(stream_compute, transfer_ready[chunk], 0));
        matmul_shared_kernel<<<grid, block, 0, stream_compute>>>(
            d_A[buffer_idx], d_B, d_C[buffer_idx], current_chunk_rows, N, K);
        CHECK_CUDA(cudaEventRecord(compute_ready[chunk], stream_compute));
        nvtxRangePop();
        
        // Stage 3: D2H Transfer (wait for compute, overlapped with next H2D and compute)
        nvtxRangePushA("D2H Transfer");
        CHECK_CUDA(cudaStreamWaitEvent(stream_d2h, compute_ready[chunk], 0));
        CHECK_CUDA(cudaMemcpyAsync(C_result + start_row * N, d_C[buffer_idx], 
                                  current_chunk_rows * N * sizeof(float), 
                                  cudaMemcpyDeviceToHost, stream_d2h));
        nvtxRangePop();
    }
    
    // Synchronize all streams
    CHECK_CUDA(cudaStreamSynchronize(stream_h2d));
    CHECK_CUDA(cudaStreamSynchronize(stream_compute));
    CHECK_CUDA(cudaStreamSynchronize(stream_d2h));
    
    nvtxRangePop();
    
    // Cleanup
    for (int i = 0; i < 3; i++) {
        CHECK_CUDA(cudaFree(d_A[i]));
        CHECK_CUDA(cudaFree(d_C[i]));
    }
    CHECK_CUDA(cudaFree(d_B));
    
    for (int i = 0; i < num_chunks; i++) {
        CHECK_CUDA(cudaEventDestroy(transfer_ready[i]));
        CHECK_CUDA(cudaEventDestroy(compute_ready[i]));
    }
    free(transfer_ready);
    free(compute_ready);
    
    CHECK_CUDA(cudaStreamDestroy(stream_h2d));
    CHECK_CUDA(cudaStreamDestroy(stream_compute));
    CHECK_CUDA(cudaStreamDestroy(stream_d2h));
}

// Original sequential version for comparison
void run_sequential_matmul(float *A, float *B, float *C_result, int M, int N, int K, 
                          bool use_shared_memory = false) {
    
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    dim3 block(use_shared_memory ? TILE_SIZE : 16, use_shared_memory ? TILE_SIZE : 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    nvtxRangePushA("Sequential Execution");
    
    // Sequential: H2D → Compute → D2H (no overlapping)
    nvtxRangePushA("H2D Transfer");
    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    nvtxRangePop();
    
    nvtxRangePushA("Compute");
    if (use_shared_memory) {
        matmul_shared_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    } else {
        matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    nvtxRangePop();
    
    nvtxRangePushA("D2H Transfer");
    CHECK_CUDA(cudaMemcpy(C_result, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();
    
    nvtxRangePop();
    
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
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
    printf("True overlapped execution with chunked processing...\n");

    // Allocate pinned host memory for faster transfers
    float *A, *B, *C_cpu, *C_sequential, *C_overlapped, *C_pipelined;
    CHECK_CUDA(cudaMallocHost(&A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_cpu, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_sequential, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_overlapped, M * N * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&C_pipelined, M * N * sizeof(float)));

    // Initialize matrices
    init_matrix(A, M * K, 1.0f);
    init_matrix(B, K * N, 1.0f);
    init_matrix(C_cpu, M * N, 0.0f);
    init_matrix(C_sequential, M * N, 0.0f);
    init_matrix(C_overlapped, M * N, 0.0f);
    init_matrix(C_pipelined, M * N, 0.0f);

    // CPU computation for reference
    printf("\n--- CPU Computation ---\n");
    nvtxRangePushA("CPU Computation");
    double start_cpu = get_time_ms();
    naive_cpu_matmul(A, B, C_cpu, M, N, K);
    double end_cpu = get_time_ms();
    nvtxRangePop();
    printf("CPU execution time: %.2f ms\n", end_cpu - start_cpu);

    // Sequential GPU execution (baseline)
    printf("\n--- GPU Sequential (Baseline) ---\n");
    double start_sequential = get_time_ms();
    run_sequential_matmul(A, B, C_sequential, M, N, K, true);
    double end_sequential = get_time_ms();
    
    bool valid_sequential = compare_matrices(C_cpu, C_sequential, M * N);
    printf("Results: %s\n", valid_sequential ? "PASS" : "FAIL");
    printf("Execution time: %.2f ms\n", end_sequential - start_sequential);
    printf("Speedup: %.2fx\n", (end_cpu - start_cpu) / (end_sequential - start_sequential));

    // True overlapped execution
    printf("\n--- GPU True Overlapped ---\n");
    double start_overlapped = get_time_ms();
    run_true_overlapped_matmul(A, B, C_overlapped, M, N, K, true);
    double end_overlapped = get_time_ms();
    
    bool valid_overlapped = compare_matrices(C_cpu, C_overlapped, M * N);
    printf("Results: %s\n", valid_overlapped ? "PASS" : "FAIL");
    printf("Execution time: %.2f ms\n", end_overlapped - start_overlapped);
    printf("Speedup vs CPU: %.2fx\n", (end_cpu - start_cpu) / (end_overlapped - start_overlapped));
    printf("Speedup vs Sequential GPU: %.2fx\n", (end_sequential - start_sequential) / (end_overlapped - start_overlapped));

    // Pipelined execution
    printf("\n--- GPU Pipelined ---\n");
    double start_pipelined = get_time_ms();
    run_pipelined_matmul(A, B, C_pipelined, M, N, K);
    double end_pipelined = get_time_ms();
    
    bool valid_pipelined = compare_matrices(C_cpu, C_pipelined, M * N);
    printf("Results: %s\n", valid_pipelined ? "PASS" : "FAIL");
    printf("Execution time: %.2f ms\n", end_pipelined - start_pipelined);
    printf("Speedup vs CPU: %.2fx\n", (end_cpu - start_cpu) / (end_pipelined - start_pipelined));
    printf("Speedup vs Sequential GPU: %.2fx\n", (end_sequential - start_sequential) / (end_pipelined - start_pipelined));

    // Performance summary
    printf("\n=== Performance Summary ===\n");
    printf("CPU Time:              %.2f ms\n", end_cpu - start_cpu);
    printf("GPU Sequential:        %.2f ms (%.2fx speedup)\n", 
           end_sequential - start_sequential,
           (end_cpu - start_cpu) / (end_sequential - start_sequential));
    printf("GPU True Overlapped:   %.2f ms (%.2fx vs sequential)\n",
           end_overlapped - start_overlapped,
           (end_sequential - start_sequential) / (end_overlapped - start_overlapped));
    printf("GPU Pipelined:         %.2f ms (%.2fx vs sequential)\n",
           end_pipelined - start_pipelined,
           (end_sequential - start_sequential) / (end_pipelined - start_pipelined));

    // Theoretical analysis
    double data_size_mb = (M * K + K * N + M * N) * sizeof(float) / (1024.0 * 1024.0);
    printf("\nOverlap Efficiency Analysis:\n");
    printf("Total data size: %.2f MB\n", data_size_mb);
    printf("Sequential effective bandwidth: %.2f GB/s\n",
           data_size_mb / (end_sequential - start_sequential));
    printf("Overlapped effective bandwidth: %.2f GB/s\n",
           data_size_mb / (end_overlapped - start_overlapped));

    // Cleanup
    CHECK_CUDA(cudaFreeHost(A));
    CHECK_CUDA(cudaFreeHost(B));
    CHECK_CUDA(cudaFreeHost(C_cpu));
    CHECK_CUDA(cudaFreeHost(C_sequential));
    CHECK_CUDA(cudaFreeHost(C_overlapped));
    CHECK_CUDA(cudaFreeHost(C_pipelined));

    return 0;
}
