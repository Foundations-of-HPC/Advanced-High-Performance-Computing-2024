#include <stdio.h>
#define epsilon (float)1e-5

// Thread block size
#define NB 32

// Forward declaration
void randomInit (float*, int);
void MatMul_cpu (const float *, const float *, float *, int );
void MatMul_gpu (const float *, const float *, float *, int );
__global__ void MatMul_kernel(float *, float *, float *, int); 


int main(int argc, char** argv) {

  // Matrix dimensions: N x N
  // Matrix dimensions are assumed to be multiples of NB 
  int N = 32*NB;

  // matrices on the host
  float *h_A, *h_B;

  // results on host
  float *cpu_result;
  float *gpu_result;

  // size in bytes
  size_t size = N*N * sizeof(float);

  // allocate matrices on the host
  h_A = (float *) malloc(size * sizeof(float));
  h_B = (float *) malloc(size * sizeof(float));

  // init matrices
  randomInit(h_A, N*N);
  randomInit(h_B, N*N);

  // allocate matrices to compare the results CPU/GPU
  cpu_result = (float *) malloc(size * sizeof(float));
  gpu_result = (float *) malloc(size * sizeof(float));


  // compute on GPU
  MatMul_gpu (h_A, h_B, gpu_result, N);

  // compute on CPU
  MatMul_cpu (h_A, h_B, cpu_result, N);


  // check results
  int error = 0;
  for(int i=0; i<N*N; i++) {
     float cpu_value = cpu_result[i];
     if(fabs(cpu_value - gpu_result[i])> epsilon*cpu_value)
	error++;
  }

  if(error==0)
    printf("\nTEST PASSED\n");
  else
    printf("\n\nTEST FAILED: number of errors:  %d\n", error);

  free(h_A);
  free(h_B);
  free(cpu_result);
  free(gpu_result);

}

// Matrices are stored in row-major order:
// M(row, col) = *(M + row * N + col) 

__device__ int get_offset (int idx_i, int idx_j, int N) {
   return idx_i * N * NB + idx_j * NB;
}

void MatMul_gpu(const float *h_A, const float *h_B, float *h_C, int N) { 
   cudaEvent_t start, stop;
   size_t size = N*N * sizeof(float);

   float *d_A, *d_B, *d_C;

// Load A and B to device memory 
   cudaMalloc((void **)&d_A, size);
   cudaMalloc((void **)&d_B, size);

   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
   cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

// Allocate C in device memory 
   cudaMalloc((void **)&d_C, size);

// Grid specify
   dim3 dimBlock (NB, NB); 
   dim3 dimGrid  (N / dimBlock.x, N / dimBlock.x);

   cudaEventCreate(&start);
   cudaEventCreate(&stop);
// Start timing
   cudaEventRecord(start);

// Invoke kernel 
   MatMul_kernel <<<dimGrid, dimBlock>>> (d_A, d_B, d_C, N);

// End timing
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   float gpu_time;
   cudaEventElapsedTime(&gpu_time, start, stop);
   double time_sec = gpu_time/1000.0;
   double num_ops = 2.0 * (double) N * (double) N * (double) N;
   double gflops = 1.0e-9 * num_ops/time_sec;
   printf("CUDA Gflops = %.4f , Time = %.5f s dim=%d\n", gflops, time_sec, N);

// Read C from device memory 
   cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); 

// Free device memory 
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

}


// Matrix multiplication kernel called by MatMul_gpu() 
__global__ void MatMul_kernel(float *A, float *B, float *C, int N) {

   // Shared memory used to store Asub and Bsub respectively
   __shared__ float As[NB][NB];
   __shared__ float Bs[NB][NB];
 
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

   for (int kb = 0; kb < (N / NB); ++kb) {

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


      // Synchronize to make sure the sub-matrices are loaded
      // before starting the computation
      // ---------------- //
      // INSERT CUDA CODE //
      // ---------------- //


      // Multiply As and Bs together
      for (int k = 0; k < NB; ++k) {
         // ---------------- //
         // INSERT CUDA CODE //
         // ---------------- //

      }
      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      // ---------------- //
      // INSERT CUDA CODE //
      // ---------------- //

   }
   
   c_offset = get_offset (ib, jb, N);
   // Each thread block computes one sub-matrix Csub of C
   // ---------------- //
   // INSERT CUDA CODE //
   // ---------------- //


}


void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void MatMul_cpu (const float *A, const float *B, float *C, int N) {

   for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
	 float value = 0.0f;
	 for (int k = 0; k < N; k++) {
	    value += A[i*N+k] * B[k*N+j];
	 }
	 C[i*N + j] = value;
      }
   }

}

