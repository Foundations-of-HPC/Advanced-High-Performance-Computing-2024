// File: helper.cuh
#pragma once

#ifndef HELPER_H
#define HELPER_H

#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

// CUDA kernel for vector addition - no stride
__global__ void vectorAddNostride(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for vector addition - with stride
__global__ void vectorAddstride(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// CPU version
void vectorAddCPU(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA error check
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#endif // HELPER_H

