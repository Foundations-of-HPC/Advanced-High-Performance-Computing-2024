#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include "helper.cuh"  // Include the helper functions


int main(int argc, char* argv[]) {
    // Vector size - adjust based on your system's memory
    
   if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <vector_size>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Error: vector_size must be a positive integer." << std::endl;
        return 1;
    }

   const size_t bytes = N * sizeof(float);
    
// Host vectors
    std::vector<float> h_a(N), h_b(N), h_c(N);
    
    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    std::cout << "Vector Addition Performance Test\n";
    std::cout << "Vector size: " << N << " elements (" << bytes/1024/1024 << " MB per vector)\n\n";
    
    // Test 1: Single Thread (GPU - baseline)
    CHECK_CUDA(cudaEventRecord(start));
    vectorAddSingleThread<<<1, 1>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float single_thread_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&single_thread_time_ms, start, stop));
    long long single_thread_time_ns = (long long)(single_thread_time_ms * 1e6);
    
    double single_thread_bandwidth = (3.0 * bytes) / (single_thread_time_ns / 1e9) / 1e6; // MB/s
    
    std::cout << "Single Thread (GPU):\n";
    std::cout << "  Time: " << single_thread_time_ns << " ns\n";
    std::cout << "  Speedup: 1x\n";
    std::cout << "  Bandwidth: " << std::round(single_thread_bandwidth) << " MB/s\n\n";
    
    // Test 2: Single Block (256 threads)
    const int threadsPerBlock = 256;
    
    CHECK_CUDA(cudaEventRecord(start));
    vectorAddSingleBlock<<<1, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float gpu_single_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_single_time_ms, start, stop));
    long long gpu_single_time_ns = (long long)(gpu_single_time_ms * 1e6);
    
    double gpu_single_speedup = (double)single_thread_time_ns / gpu_single_time_ns;
    double gpu_single_bandwidth = (3.0 * bytes) / (gpu_single_time_ns / 1e9) / 1e9;
    
    std::cout << "Single Block (256 threads):\n";
    std::cout << "  Time: " << gpu_single_time_ns << " ns\n";
    std::cout << "  Speedup: " << std::round(gpu_single_speedup) << "x\n";
    std::cout << "  Bandwidth: " << std::round(gpu_single_bandwidth) << " GB/s\n\n";
    
    // Test 3: Multiple Blocks
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    
    // Calculate optimal grid size
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    numBlocks = std::min(numBlocks, prop.multiProcessorCount * 32); // Limit to reasonable number
    
    CHECK_CUDA(cudaEventRecord(start));
    vectorAddMultipleBlocks<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float gpu_multi_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&gpu_multi_time_ms, start, stop));
    long long gpu_multi_time_ns = (long long)(gpu_multi_time_ms * 1e6);
    
    double gpu_multi_speedup = (double)single_thread_time_ns / gpu_multi_time_ns;
    double gpu_multi_bandwidth = (3.0 * bytes) / (gpu_multi_time_ns / 1e9) / 1e9;
    
    std::cout << "Multiple Blocks (" << numBlocks << " blocks, " << threadsPerBlock << " threads each):\n";
    std::cout << "  Time: " << gpu_multi_time_ns << " ns\n";
    std::cout << "  Speedup: " << std::round(gpu_multi_speedup) << "x\n";
    std::cout << "  Bandwidth: " << std::round(gpu_multi_bandwidth) << " GB/s\n\n";
    
    // Verify results
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < std::min(1000, N); i++) {
        if (std::abs(h_c[i] - 3.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Results verification: " << (correct ? "PASSED" : "FAILED") << "\n";
    
    // Print device info
    std::cout << "\nGPU Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    return 0;
}
