#include <stdio.h>
#include <assert.h>
#include <nvtx3/nvToolsExt.h>

// CUDA kernel for vector addition - multiple blocks version
__global__ void vectorAddMultipleBlocks(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// Convenience function for checking CUDA runtime API results
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void profileVectorAdd(float        *h_a, 
                     float        *h_b, 
                     float        *h_c,
                     float        *d_a,
                     float        *d_b,
                     float        *d_c,
                     unsigned int  n,
                     const char   *desc)
{
    printf("\n%s vector addition\n", desc);
    unsigned int bytes = n * sizeof(float);
    
    // Create NVTX range for the entire operation
    char nvtx_name[256];
    sprintf(nvtx_name, "VectorAdd_%s", desc);
    nvtxRangePushA(nvtx_name);
    
    // events for timing
    cudaEvent_t startEvent, stopEvent; 
    
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    
    // Time the entire operation including memory transfers
    checkCuda( cudaEventRecord(startEvent, 0) );
    
    // Copy input vectors to device with NVTX tags
    sprintf(nvtx_name, "MemcpyH2D_%s_VecA", desc);
    nvtxRangePushA(nvtx_name);
    checkCuda( cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice) );
    nvtxRangePop();
    
    sprintf(nvtx_name, "MemcpyH2D_%s_VecB", desc);
    nvtxRangePushA(nvtx_name);
    checkCuda( cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice) );
    nvtxRangePop();
    
    // Launch kernel with NVTX tag
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    // Limit the number of blocks to prevent excessive grid size
    numBlocks = min(numBlocks, 65535);
    
    sprintf(nvtx_name, "Kernel_%s_VectorAdd", desc);
    nvtxRangePushA(nvtx_name);
    vectorAddMultipleBlocks<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
    nvtxRangePop();
    
    // Copy result back to host with NVTX tag
    sprintf(nvtx_name, "MemcpyD2H_%s_Result", desc);
    nvtxRangePushA(nvtx_name);
    checkCuda( cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost) );
    nvtxRangePop();
    
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    
    float time;
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  Total time (ms): %f\n", time);
    printf("  Effective bandwidth (GB/s): %f\n", (3 * bytes * 1e-6) / time);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < n && i < 1000; ++i) { // Check first 1000 elements
        if (abs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            printf("*** %s vector addition failed at index %d: %f + %f != %f ***\n", 
                   desc, i, h_a[i], h_b[i], h_c[i]);
            correct = false;
            break;
        }
    }
    if (correct) {
        printf("  Vector addition successful!\n");
    }
    
    // clean up events
    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );
    
    // End NVTX range for entire operation
    nvtxRangePop();
}

void profileCopies(float        *h_a, 
                   float        *h_b, 
                   float        *d, 
                   unsigned int  n,
                   const char   *desc)
{
    printf("\n%s transfers\n", desc);
    unsigned int bytes = n * sizeof(float);
    
    // Create NVTX range for memory transfer profiling
    char nvtx_name[256];
    sprintf(nvtx_name, "MemTransfer_%s", desc);
    nvtxRangePushA(nvtx_name);
    
    // events for timing
    cudaEvent_t startEvent, stopEvent; 
    
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    
    // Host to Device transfer with NVTX tag
    sprintf(nvtx_name, "H2D_Transfer_%s", desc);
    nvtxRangePushA(nvtx_name);
    checkCuda( cudaEventRecord(startEvent, 0) );
    checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    nvtxRangePop();
    
    float time;
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);
    
    // Device to Host transfer with NVTX tag
    sprintf(nvtx_name, "D2H_Transfer_%s", desc);
    nvtxRangePushA(nvtx_name);
    checkCuda( cudaEventRecord(startEvent, 0) );
    checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
    checkCuda( cudaEventRecord(stopEvent, 0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    nvtxRangePop();
    
    checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
    printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);
    
    for (int i = 0; i < n; ++i) {
        if (h_a[i] != h_b[i]) {
            printf("*** %s transfers failed ***", desc);
            break;
        }
    }
    
    // clean up events
    checkCuda( cudaEventDestroy(startEvent) );
    checkCuda( cudaEventDestroy(stopEvent) );
    
    // End NVTX range for memory transfers
    nvtxRangePop();
}

int main()
{
    unsigned int nElements = 4*1024*1024;
    const unsigned int bytes = nElements * sizeof(float);
    
    // host arrays for memory transfer testing
    float *h_aPageable, *h_bPageable;   
    float *h_aPinned, *h_bPinned;
    
    // host arrays for vector addition
    float *h_a_vecPageable, *h_b_vecPageable, *h_c_vecPageable;
    float *h_a_vecPinned, *h_b_vecPinned, *h_c_vecPinned;
    
    // device arrays
    float *d_a, *d_b, *d_c;
    
    // allocate and initialize memory transfer arrays
    h_aPageable = (float*)malloc(bytes);                    // host pageable
    h_bPageable = (float*)malloc(bytes);                    // host pageable
    checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // host pinned
    checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // host pinned
    
    // allocate vector addition arrays
    h_a_vecPageable = (float*)malloc(bytes);
    h_b_vecPageable = (float*)malloc(bytes);
    h_c_vecPageable = (float*)malloc(bytes);
    checkCuda( cudaMallocHost((void**)&h_a_vecPinned, bytes) );
    checkCuda( cudaMallocHost((void**)&h_b_vecPinned, bytes) );
    checkCuda( cudaMallocHost((void**)&h_c_vecPinned, bytes) );
    
    // allocate device memory
    checkCuda( cudaMalloc((void**)&d_a, bytes) );           // device
    checkCuda( cudaMalloc((void**)&d_b, bytes) );           // device  
    checkCuda( cudaMalloc((void**)&d_c, bytes) );           // device
    
    // initialize arrays
    for (int i = 0; i < nElements; ++i) {
        h_aPageable[i] = i;
        h_a_vecPageable[i] = i;
        h_b_vecPageable[i] = i * 2.0f;
    }
    
    memcpy(h_aPinned, h_aPageable, bytes);
    memcpy(h_a_vecPinned, h_a_vecPageable, bytes);
    memcpy(h_b_vecPinned, h_b_vecPageable, bytes);
    
    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);
    memset(h_c_vecPageable, 0, bytes);
    memset(h_c_vecPinned, 0, bytes);
    
    // output device info and transfer size
    cudaDeviceProp prop;
    checkCuda( cudaGetDeviceProperties(&prop, 0) );
    printf("\nDevice: %s\n", prop.name);
    printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));
    printf("Number of elements: %d\n", nElements);
    
    // perform memory bandwidth tests
    profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
    
    // perform vector addition tests
    profileVectorAdd(h_a_vecPageable, h_b_vecPageable, h_c_vecPageable, 
                     d_a, d_b, d_c, nElements, "Pageable");
    profileVectorAdd(h_a_vecPinned, h_b_vecPinned, h_c_vecPinned, 
                     d_a, d_b, d_c, nElements, "Pinned");
    
    printf("\n");
    
    // cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    cudaFreeHost(h_a_vecPinned);
    cudaFreeHost(h_b_vecPinned);
    cudaFreeHost(h_c_vecPinned);
    free(h_aPageable);
    free(h_bPageable);
    free(h_a_vecPageable);
    free(h_b_vecPageable);
    free(h_c_vecPageable);
    
    return 0;
}
