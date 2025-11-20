#pragma once

#ifndef HELPER_CUH
#define HELPER_CUH

#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>

// Maximum number of dimensions (can be up to 3)
#define MAX_DIM 3

#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

void initializeData(float *data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }
}

float calculateBandwidth(int nx, int ny, float elapsed) {
    return (2.0f * nx * ny * sizeof(float)) / (elapsed * 1e6);  // GB/s
}

__global__ void copyRow(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy*nx + ix] = in[iy*nx + ix];
    }
}

__global__ void copyCol(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[ix*ny + iy] = in[ix*ny + iy];
    }
}

__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[iy*nx + ix] = in[ix*ny + iy];
    }
}

__global__ void transposeUnroll4Row(float *out, float *in, const int nx, const int ny) {
	unsigned int ix = blockDim.x * blockIdx.x*4 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int ti = iy*nx + ix; unsigned int to = ix*ny + iy; // access in rows
	
	// access in columns
	if (ix+3*blockDim.x < nx && iy < ny) {
		out[to] = in[ti];
		out[to + ny*blockDim.x] = in[ti+blockDim.x];
		out[to + ny*2*blockDim.x] = in[ti+2*blockDim.x];
		out[to + ny*3*blockDim.x] = in[ti+3*blockDim.x];
	}
}

__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny) {
	unsigned int ix = blockDim.x * blockIdx.x*4 + threadIdx.x;
	unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int ti = iy*nx + ix; unsigned int to = ix*ny + iy; // access in rows

	// access in columns
	if (ix+3*blockDim.x < nx && iy < ny) {
		out[ti] = in[to];
		out[ti + blockDim.x] = in[to+ blockDim.x*ny];
		out[ti + 2*blockDim.x] = in[to+ 2*blockDim.x*ny];
		out[ti + 3*blockDim.x] = in[to+ 3*blockDim.x*ny];
	}
}

#endif // HELPER_CUH

