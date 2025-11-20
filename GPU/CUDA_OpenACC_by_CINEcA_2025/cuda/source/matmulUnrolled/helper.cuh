#pragma once

// File: helper.cuh
#ifndef HELPER_H
#define HELPER_H

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

// CUDA kernel for vector addition - single thread on GPU
__global__ void vectorAddSingleThread(float* a, float* b, float* c, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }
}

// CUDA kernel for vector addition - single block version
__global__ void vectorAddSingleBlock(float* a, float* b, float* c, int n) {
    int idx = threadIdx.x;
    int stride = blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition - multiple blocks version
__global__ void vectorAddMultipleBlocks(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// CPU single-threaded version
void vectorAddCPU(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Utility function to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#endif // HELPER_H
