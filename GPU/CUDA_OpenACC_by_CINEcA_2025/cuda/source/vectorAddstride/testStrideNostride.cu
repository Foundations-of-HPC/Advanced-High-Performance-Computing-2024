#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Kernel 1: No stride (each thread handles one element)
__global__ void vectorAddNostride(float* a, float* b, float* c, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel 2: Strided access (each thread handles multiple elements with stride)
__global__ void vectorAddstride(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Utility to initialize host data
void initVectors(float* a, float* b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }
}

// Utility to validate result
bool validate(float* a, float* b, int n) {
    for (int i = 0; i < n; ++i) {
        if (abs(a[i] - b[i]) > 1e-5f) {
            return false;
        }
    }
    return true;
}

// Run and time a kernel
void runKernel(bool useStride, int n, int threadsPerBlock, int blocks) {
    std::cout << "\n=== Running with " << (useStride ? "Strided" : "No-Stride") << " Kernel ===" << std::endl;

    size_t bytes = n * sizeof(float);
    float *h_a, *h_b, *h_c;

    h_a = new float[n];
    h_b = new float[n];
    h_c = new float[n];

    initVectors(h_a, h_b, n);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    auto start = std::chrono::high_resolution_clock::now();
    if (useStride)
        vectorAddstride<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    else
        vectorAddNostride<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Validation
    bool valid = true;
    for (int i = 0; i < n; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            valid = false;
            std::cout << "Mismatch at index " << i << std::endl;
            break;
        }
    }

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "Validation: " << (valid ? "PASSED" : "FAILED") << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    // Parameters
    int threadsPerBlock = 256;

    // Case 1: Small vector
    std::cout << "\n--- Case 1: Small Vector (n = 1 << 16) ---" << std::endl;
    runKernel(false, 1 << 16, threadsPerBlock, 256);  // Enough threads
    runKernel(true,  1 << 16, threadsPerBlock, 64);   // Fewer blocks

    // Case 2: Large vector, enough threads
    std::cout << "\n--- Case 2: Large Vector, Enough Threads (n = 1 << 24) ---" << std::endl;
    runKernel(false, 1 << 24, threadsPerBlock, (1 << 24) / threadsPerBlock);
    runKernel(true,  1 << 24, threadsPerBlock, (1 << 24) / threadsPerBlock);

    // Case 3: Large vector, limited threads
    std::cout << "\n--- Case 3: Large Vector, Limited Threads (n = 1 << 24) ---" << std::endl;
    runKernel(false, 1 << 24, threadsPerBlock, 64);   // Not enough threads
    runKernel(true,  1 << 24, threadsPerBlock, 64);   // Stride helps here

    return 0;
}

