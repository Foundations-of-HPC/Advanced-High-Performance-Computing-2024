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

        // Load tile of B into shared memory (transposed for coalesced access)
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

    // Allocate and initialize host matrices
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C_cpu = (float*)malloc(M * N * sizeof(float));
    float *C_gpu = (float*)malloc(M * N * sizeof(float));

    init_matrix(A, M * K, 1.0f);   // Initialize with 1.0
    init_matrix(B, K * N, 1.0f);   // Initialize with 1.0
    init_matrix(C_cpu, M * N, 0.0f);
    init_matrix(C_gpu, M * N, 0.0f);

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Create CUDA events for timing kernel
    cudaEvent_t kernel_start, kernel_stop;
    CHECK_CUDA(cudaEventCreate(&kernel_start));
    CHECK_CUDA(cudaEventCreate(&kernel_stop));

    // Configure kernel with shared memory
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // ================= GPU Execution =================
    nvtxRangePushA("Total GPU Time");
    double start_gpu_total = get_time_ms();
    
    // Memory transfer: Host to Device
    nvtxRangePushA("H2D Transfer");
    CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
    nvtxRangePop();  // End H2D Transfer
    
    // Kernel execution with shared memory
    nvtxRangePushA("MatMul Kernel (Shared Memory)");
    CHECK_CUDA(cudaEventRecord(kernel_start));
    matmul_shared_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA(cudaEventRecord(kernel_stop));
    CHECK_CUDA(cudaEventSynchronize(kernel_stop));
    nvtxRangePop();  // End Kernel
    
    // Memory transfer: Device to Host
    nvtxRangePushA("D2H Transfer");
    CHECK_CUDA(cudaMemcpy(C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();  // End D2H Transfer
    
    nvtxRangePop();  // End Total GPU Time
    double end_gpu_total = get_time_ms();

    // ================= CPU Computation =================
    nvtxRangePushA("CPU Computation");
    double start_cpu = get_time_ms();
    naive_cpu_matmul(A, B, C_cpu, M, N, K);
    double end_cpu = get_time_ms();
    nvtxRangePop();  // End CPU Computation

    // Calculate elapsed times
    float kernel_time;
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop));
    double gpu_total_time = end_gpu_total - start_gpu_total;
    double cpu_time = end_cpu - start_cpu;

    // Verify results
    bool valid = compare_matrices(C_cpu, C_gpu, M * N);

    // Print results
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Tile size: %dx%d\n", TILE_SIZE, TILE_SIZE);
    printf("CPU execution time: %.2f ms\n", cpu_time);
    printf("GPU Results: %s\n", valid ? "PASS" : "FAIL");
    printf("  Kernel time: %.2f ms\n", kernel_time);
    printf("  Total GPU time (incl. transfer): %.2f ms\n", gpu_total_time);
    printf("  Speedup (kernel): %.2fx\n", cpu_time / kernel_time);
    printf("  Speedup (total): %.2fx\n", cpu_time / gpu_total_time);

    // Cleanup
    free(A); free(B); free(C_cpu); free(C_gpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(kernel_start));
    CHECK_CUDA(cudaEventDestroy(kernel_stop));

    return 0;
}

