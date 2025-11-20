#include <math.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>

#include <cub/device/device_reduce.cuh>
using namespace cub;

#define NUM_THREADS 32


__global__ void jacobi(float * curr, float * next, float * errs, int n, int stride){

    // INSERT CODE: index definitions
    int i = ;
    int j = ;

    int idx = ; 

    // INSERT CODE: 
    if ( ) {

    // INSERT CODE: 
        next[idx] = 0.25 * ( ) ;

        errs[idx] = fabs(next[idx] - curr[idx]);

    }

}


int main(int argc, char *argv[]) {

    int n = 4096;
    int stride;
    int iter_max = 1000;
    double tol = 1.0e-6;

    if (argc > 1) {
        n = std::stoi(argv[1]);
    }

    if (argc > 2) {
        iter_max = std::stoi(argv[2]);
    }

    stride = n;

    std::cout << "Dim: " << n << std::endl;
    std::cout << "Stride: " << stride << std::endl;

    size_t nbytes = sizeof(float) * n * stride;

    float * h_curr = (float *) malloc(nbytes);

    float h_err = 1.;

    float * d_curr;
    float * d_next;
    float * d_errs;

    float * d_err;

    cudaMalloc((void **) &d_curr, nbytes);
    cudaMalloc((void **) &d_next, nbytes);
    cudaMalloc((void **) &d_errs, nbytes);

    cudaMalloc((void **) &d_err, sizeof(float));

    cudaMemset(d_curr, 0, nbytes);
    cudaMemset(d_next, 0, nbytes);

    dim3 nThreads(NUM_THREADS, NUM_THREADS);
    dim3 nBlocks((n-1)/nThreads.x+1, (n-1)/nThreads.y+1);

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_errs, d_err, n*stride);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // timing
    cudaEvent_t start, stop;
    float etime, time_fd, time_max;

    time_fd = 0.0;
    time_max = 0.0;

    // create events
    // INSERT CODE: 

    int iter = 0;
    struct timeval temp_1, temp_2;
    double ttime=0.;

    for ( int i = 0 ; i < n ; i++ ) {
        h_curr[i * stride] = 1.;
    }

    cudaMemcpy(d_curr, h_curr, nbytes, cudaMemcpyHostToDevice);

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, n);

    gettimeofday(&temp_1, (struct timezone*)0);

    while ( h_err > tol && iter < iter_max )
    {

        cudaEventRecord(start);

        jacobi<<<nBlocks, nThreads>>>(d_curr, d_next, d_errs, n, stride);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&etime, start, stop);

        time_fd += etime;

        cudaEventRecord(start);

        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_errs, d_err, n*stride);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&etime, start, stop);

        time_max += etime;

        cudaMemcpy(&h_err, d_err, sizeof(float), cudaMemcpyDeviceToHost);

        //if(iter % 10 == 0) printf("%5d, %0.8lf\n", iter, h_err);

        std::swap(d_curr, d_next);

        iter++;
    }

    gettimeofday(&temp_2, (struct timezone*)0);
    ttime = 0.000001*((temp_2.tv_sec-temp_1.tv_sec)*1.e6+(temp_2.tv_usec-temp_1 .tv_usec));

    printf("Elapsed time (s) = %.2lf\n", ttime);
    printf("Stopped at iteration: %u\n", iter);

    time_fd /= iter;
    time_max /= iter;

    std::cout << "Time FD  / ms " << time_fd << std::endl;
    std::cout << "Time MAX / ms " << time_max << std::endl;

    free(h_curr);

    cudaFree(d_curr);
    cudaFree(d_next);

    // release event resources
    // INSERT CODE: 

    return 0;

}
