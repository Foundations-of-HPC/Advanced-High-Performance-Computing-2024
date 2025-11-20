#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define TILE_SIZE 16
#define MAX_THREADS 8

// Structure to pass data to thread function
typedef struct {
    float *A;
    float *B;
    float *C;
    int N;
    int start_row;
    int end_row;
    int thread_id;
} ThreadData;

// Basic matrix multiplication
void matrix_mult_basic(float *A, float *B, float *C, int N) {
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

// Optimized matrix multiplication with loop reordering (cache-friendly)
void matrix_mult_optimized(float *A, float *B, float *C, int N) {
    // Initialize result matrix to zero
    memset(C, 0, N * N * sizeof(float));
    
    // Loop reordering for better cache performance (ikj order)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// Blocked matrix multiplication (tiled approach)
void matrix_mult_blocked(float *A, float *B, float *C, int N) {
    // Initialize result matrix to zero
    memset(C, 0, N * N * sizeof(float));
    
    // Block/tile size
    int block_size = TILE_SIZE;
    
    for (int ii = 0; ii < N; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < N; kk += block_size) {
                // Compute the actual block boundaries
                int i_end = (ii + block_size < N) ? ii + block_size : N;
                int j_end = (jj + block_size < N) ? jj + block_size : N;
                int k_end = (kk + block_size < N) ? kk + block_size : N;
                
                // Multiply the blocks
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        float a_ik = A[i * N + k];
                        for (int j = jj; j < j_end; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// Thread function for parallel matrix multiplication
void* thread_matrix_mult(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    
    // Each thread computes a subset of rows
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < data->N; k++) {
                sum += data->A[i * data->N + k] * data->B[k * data->N + j];
            }
            data->C[i * data->N + j] = sum;
        }
    }
    
    return NULL;
}

// Multi-threaded matrix multiplication
void matrix_mult_parallel(float *A, float *B, float *C, int N, int num_threads) {
    pthread_t threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS];
    
    // Limit number of threads
    if (num_threads > MAX_THREADS) num_threads = MAX_THREADS;
    
    int rows_per_thread = N / num_threads;
    int remaining_rows = N % num_threads;
    
    // Create threads
    for (int t = 0; t < num_threads; t++) {
        thread_data[t].A = A;
        thread_data[t].B = B;
        thread_data[t].C = C;
        thread_data[t].N = N;
        thread_data[t].thread_id = t;
        
        // Calculate row range for this thread
        thread_data[t].start_row = t * rows_per_thread;
        thread_data[t].end_row = (t + 1) * rows_per_thread;
        
        // Add remaining rows to the last thread
        if (t == num_threads - 1) {
            thread_data[t].end_row += remaining_rows;
        }
        
        pthread_create(&threads[t], NULL, thread_matrix_mult, &thread_data[t]);
    }
    
    // Wait for all threads to complete
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

// Validation function
int validate_result(float *result1, float *result2, int N) {
    float tolerance = 1e-3f;
    int errors = 0;
    
    for (int i = 0; i < N * N; i++) {
        if (fabs(result1[i] - result2[i]) > tolerance) {
            errors++;
            if (errors <= 10) { // Print first 10 errors
                printf("Error at index %d: A=%.6f, B=%.6f, diff=%.6f\n", 
                       i, result1[i], result2[i], fabs(result1[i] - result2[i]));
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

// Get number of CPU cores
int get_cpu_cores() {
    int cores = sysconf(_SC_NPROCESSORS_ONLN);
    return cores > 0 ? cores : 1;
}

int main() {
    printf("=== C Matrix Multiplication Performance Comparison ===\n\n");
    
    // Get system information
    int cpu_cores = get_cpu_cores();
    printf("System CPU cores: %d\n", cpu_cores);
    printf("Using %d threads for parallel computation\n\n", cpu_cores);
    
    // Problem sizes to test
    int sizes[] = {2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        printf("Problem Size: %d x %d\n", N, N);
        printf("Matrix elements: %d\n", N * N);
        
        size_t bytes = N * N * sizeof(float);
        printf("Memory per matrix: %.2f MB\n", bytes / (1024.0 * 1024.0));
        printf("Total memory: %.2f MB\n", 5.0 * bytes / (1024.0 * 1024.0));
        
        // Allocate memory for matrices
        float *A = (float*)malloc(bytes);
        float *B = (float*)malloc(bytes);
        float *C_basic = (float*)malloc(bytes);
        float *C_optimized = (float*)malloc(bytes);
        float *C_blocked = (float*)malloc(bytes);
        float *C_parallel = (float*)malloc(bytes);
        
        if (!A || !B || !C_basic || !C_optimized || !C_blocked || !C_parallel) {
            printf("Failed to allocate memory!\n");
            return 1;
        }
        
        // Initialize matrices
        init_matrix(A, N);
        init_matrix(B, N);
        
        // Test 1: Basic matrix multiplication
        printf("\n=== Basic Matrix Multiplication ===\n");
        double start_time = get_time();
        matrix_mult_basic(A, B, C_basic, N);
        double basic_time = get_time() - start_time;
        printf("Basic multiplication time: %.3f seconds\n", basic_time);
        
        // Test 2: Optimized matrix multiplication (cache-friendly)
        printf("\n=== Optimized Matrix Multiplication ===\n");
        start_time = get_time();
        matrix_mult_optimized(A, B, C_optimized, N);
        double optimized_time = get_time() - start_time;
        printf("Optimized multiplication time: %.3f seconds\n", optimized_time);
        printf("Speedup vs Basic: %.2fx\n", basic_time / optimized_time);
        
        // Test 3: Blocked matrix multiplication
        printf("\n=== Blocked Matrix Multiplication ===\n");
        start_time = get_time();
        matrix_mult_blocked(A, B, C_blocked, N);
        double blocked_time = get_time() - start_time;
        printf("Blocked multiplication time: %.3f seconds\n", blocked_time);
        printf("Speedup vs Basic: %.2fx\n", basic_time / blocked_time);
        
        // Test 4: Parallel matrix multiplication
        printf("\n=== Parallel Matrix Multiplication ===\n");
        start_time = get_time();
        matrix_mult_parallel(A, B, C_parallel, N, cpu_cores);
        double parallel_time = get_time() - start_time;
        printf("Parallel multiplication time: %.3f seconds\n", parallel_time);
        printf("Speedup vs Basic: %.2fx\n", basic_time / parallel_time);
        
        // Performance Analysis
        printf("\n=== PERFORMANCE ANALYSIS ===\n");
        double flops = 2.0 * N * N * N; // 2N^3 operations
        printf("Total FLOPs: %.2e\n", flops);
        
        printf("\nPerformance (GFLOPS):\n");
        printf("Basic:     %.2f GFLOPS\n", flops / basic_time / 1e9);
        printf("Optimized: %.2f GFLOPS\n", flops / optimized_time / 1e9);
        printf("Blocked:   %.2f GFLOPS\n", flops / blocked_time / 1e9);
        printf("Parallel:  %.2f GFLOPS\n", flops / parallel_time / 1e9);
        
        printf("\nSpeedup Summary:\n");
        printf("Optimized vs Basic: %.2fx\n", basic_time / optimized_time);
        printf("Blocked vs Basic:   %.2fx\n", basic_time / blocked_time);
        printf("Parallel vs Basic:  %.2fx\n", basic_time / parallel_time);
        
        // Find best performer
        double best_time = basic_time;
        const char* best_method = "Basic";
        
        if (optimized_time < best_time) {
            best_time = optimized_time;
            best_method = "Optimized";
        }
        if (blocked_time < best_time) {
            best_time = blocked_time;
            best_method = "Blocked";
        }
        if (parallel_time < best_time) {
            best_time = parallel_time;
            best_method = "Parallel";
        }
        
        printf("✓ Best performer: %s (%.3f seconds)\n", best_method, best_time);
        
        // Validation
        printf("\n=== VALIDATION ===\n");
        int errors_opt = validate_result(C_basic, C_optimized, N);
        int errors_block = validate_result(C_basic, C_blocked, N);
        int errors_par = validate_result(C_basic, C_parallel, N);
        
        printf("Optimized validation: %s (%d errors out of %d elements)\n", 
               errors_opt == 0 ? "PASSED" : "FAILED", errors_opt, N*N);
        printf("Blocked validation: %s (%d errors out of %d elements)\n", 
               errors_block == 0 ? "PASSED" : "FAILED", errors_block, N*N);
        printf("Parallel validation: %s (%d errors out of %d elements)\n", 
               errors_par == 0 ? "PASSED" : "FAILED", errors_par, N*N);
        
        if (errors_opt == 0 && errors_block == 0 && errors_par == 0) {
            printf("✓ All optimizations produce correct results!\n");
        }
        
        // Cleanup
        free(A);
        free(B);
        free(C_basic);
        free(C_optimized);
        free(C_blocked);
        free(C_parallel);
        
        if (s < num_sizes - 1) {
            printf("\n=================================================\n\n");
        } else {
            printf("\n");
        }
    }
    
    // Print system information
    printf("\n=== SYSTEM INFORMATION ===\n");
    printf("CPU Cores: %d\n", cpu_cores);
    printf("Tile Size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Max Threads: %d\n", MAX_THREADS);
    
    // Print optimization techniques used
    printf("\n=== OPTIMIZATION TECHNIQUES ===\n");
    printf("1. Basic: Standard triple-nested loop (ijk order)\n");
    printf("2. Optimized: Loop reordering (ikj order) for better cache locality\n");
    printf("3. Blocked: Tiled multiplication to improve cache performance\n");
    printf("4. Parallel: Multi-threaded using pthreads\n");
    
    return 0;
}
