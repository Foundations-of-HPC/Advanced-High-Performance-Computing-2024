#include <stdio.h>
#include <assert.h>
#define epsilon (float)1e-5

#define NUM_THREADS 16

void matrixMulHost(float* A, float* B, float* C, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      float pvalue = 0;
      for (int k = 0; k < n; ++k) {
        float a = A[i * n + k];
        float b = B[k * n + j];
        pvalue += a * b;
      }
      C[i * n + j] = pvalue;
    }
  }
}

__global__ void MatrixMulKernel(float* A, float* B, float* C, int n) {

   // INSERT CODE: index definitions
   int j = ;
   int i = ;

   // INSERT CODE: 
   if ( ) {

   }
}

int main(int argc, char** argv) {

  int n;

  float *h_A, *h_B, *h_C;
  float *h_ref;

  if(argc<2) {
    fprintf(stderr,"Usage: %s Width\n",argv[0]);
    exit(1);
  }

  n=atoi(argv[1]);

  if(n<1) {
    fprintf(stderr,"Error Width=%d, must be > 0\n",n);
    exit(1);
  }

  size_t nbytes = n * n * sizeof(float);

  h_A = (float *) malloc(nbytes);
  h_B = (float *) malloc(nbytes);
  h_C = (float *) malloc(nbytes);

  h_ref = (float *) malloc(nbytes);

  memset(h_C, 0, nbytes);
  memset(h_ref, 0, nbytes);

  for(int y = 0; y < n; y++){
    for(int x = 0; x < n; x++) {
      h_A[y * n + x]=(float)(((y + 1) * n + x + 1)/n);
      h_B[y * n + x]=(float)(((y + 1) * n + x + 1)/n);
    }
  }

  float flops;
  float *d_A, *d_B, *d_C;

  // CUDA grid management
  dim3 threads(NUM_THREADS, NUM_THREADS);
  dim3 blocks(n/NUM_THREADS,n/NUM_THREADS);

  cudaMalloc(&d_A, nbytes);
  cudaMalloc(&d_B, nbytes);
  cudaMalloc(&d_C, nbytes);

  cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nbytes, cudaMemcpyHostToDevice);

  // cudaGetLastError call to reset previous CUDA errors
  // INSERT CODE

  // Create start and stop CUDA events 
  // INSERT CODE



  // kernel launch
  MatrixMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, n);

  // event record, synchronization, elapsed time and destruction
  // INSERT CODE

  // device synchronization and cudaGetLastError call
  // INSERT CODE

  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed = elapsed/1000.f; // convert to seconds

  // calculate Mflops
  // INSERT CODE
  printf("Kernel elapsed time %fs \n", elapsed);
  printf("Gflops: %f\n", flops);

  // copy back results from device
  cudaMemcpy(h_C, d_C, nbytes, cudaMemcpyDeviceToHost);

  // free memory on device
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  matrixMulHost(h_A, h_B, h_ref, n);

  int errCnt = 0;
  for(int y = 0; y < n; y++){
    for(int x = 0; x < n; x++) {
      float it = h_ref[y * n + x];
      if(fabs(it - h_C[y * n + x]) > epsilon * it) {
        printf("failing x=%d, y=%d: %f!=%f \n", x, y, it, h_C[y * n + x]);
        errCnt++;
      }
    }
  }

  if(errCnt==0)
    printf("\nTEST PASSED\n");
  else
    printf("\n\nTEST FAILED: number of errors:  %d\n", errCnt);

}
