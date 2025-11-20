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
   int j = blockDim.x * blockIdx.x + threadIdx.x;
   int i = blockDim.y * blockIdx.y + threadIdx.y;

   // INSERT CODE: fastest index to threads along x direction
   if (i < n && j < n) {
      float pvalue = 0;
      for (int k = 0; k < n; ++k) {
        float a = A[i * n + k];
        float b = B[k * n + j];
        pvalue += a * b;
      }
      C[i * n + j] = pvalue;
   }
}

__device__ int get_offset (int idx_i, int idx_j, int N) {
   return idx_i * N * NUM_THREADS + idx_j * NUM_THREADS;
}

__global__ void MatrixMulKernelShared(float *A, float *B, float *C, int N) {

   // Shared memory used to store Asub and Bsub respectively
   __shared__ float As[NUM_THREADS][NUM_THREADS];
   __shared__ float Bs[NUM_THREADS][NUM_THREADS];
 
   // Block row and column 
   int ib = blockIdx.y;
   int jb = blockIdx.x;

   // Thread row and column within Csub 
   int it = threadIdx.y;
   int jt = threadIdx.x;

   int a_offset, b_offset, c_offset;

   // Each thread computes one element of Csub
   // by accumulating results into Cvalue
   float Cvalue = 0.0f;

   // Loop over all the sub-matrices of A and B that are 
   // required to compute Csub. 
   // Multiply each pair of sub-matrices together
   // and accumulate the results. 

   for (int kb = 0; kb < (N / NUM_THREADS); ++kb) {

      // Get the starting address (a_offset) of Asub
      // (sub-matrix of A of dimension NB x NB)
      // Asub is located i_block sub-matrices to the right and
      // k_block sub-matrices down from the upper-left corner of A
      a_offset = get_offset (ib, kb, N);
      // Get the starting address (b_offset) of Bsub
      b_offset = get_offset (kb, jb, N);

      // Load Asub and Bsub from device memory to shared memory
      // Each thread loads one element of each sub-matrix
      // ---------------- //
      // INSERT CUDA CODE //
      // ---------------- //
      As[it][jt] = A[a_offset + it*N + jt];
      Bs[it][jt] = B[b_offset + it*N + jt];

      // Synchronize to make sure the sub-matrices are loaded
      // before starting the computation
      // ---------------- //
      // INSERT CUDA CODE //
      // ---------------- //
      __syncthreads();

      // Multiply As and Bs together
      for (int k = 0; k < NUM_THREADS; ++k) {
         // ---------------- //
         // INSERT CUDA CODE //
         // ---------------- //
         Cvalue += As[it][k] * Bs[k][jt]; 
      }
      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      // ---------------- //
      // INSERT CUDA CODE //
      // ---------------- //
      __syncthreads();
   }
   
   c_offset = get_offset (ib, jb, N);
   // Each thread block computes one sub-matrix Csub of C
   // ---------------- //
   // INSERT CUDA CODE //
   // ---------------- //
   C[c_offset + it * N + jt] = Cvalue;

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
  cudaError_t mycudaerror  ;
  mycudaerror = cudaGetLastError()  ;

  // Create start and stop CUDA events 
  // INSERT CODE
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  // kernel launch
  MatrixMulKernelShared<<<blocks, threads>>>(d_A, d_B, d_C, n);

  // event record, synchronization, elapsed time and destruction
  // INSERT CODE
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // device synchronization and cudaGetLastError call
  // INSERT CODE
  mycudaerror = cudaGetLastError() ;
  if(mycudaerror != cudaSuccess)  {
    fprintf(stderr,"%s\n",cudaGetErrorString(mycudaerror)) ;
    exit(1);
  }

  float elapsed;
  cudaEventElapsedTime(&elapsed, start, stop);
  elapsed = elapsed/1000.f; // convert to seconds

  // calculate Mflops
  // INSERT CODE
  printf("Kernel elapsed time %fs \n", elapsed);
  flops = 2.*n*n*n ;
  flops = flops/(1.e9*elapsed) ;
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
