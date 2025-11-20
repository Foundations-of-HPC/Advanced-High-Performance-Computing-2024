
#define SIZE         512
#define NUM_THREADS   32
#define NUM_REPS       3

#include <stdio.h>


__global__ void copy(float *odata, float* idata, int width, int height)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  int index  = i + j * width;

  odata[index] = idata[index];
}


// transpose naive

__global__ void transposeNaive(float *odata, float* idata, int width, int height)
{
  int xIndex = blockIdx.x * NUM_THREADS + threadIdx.x;
  int yIndex = blockIdx.y * NUM_THREADS + threadIdx.y;

  int index_in  = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;

  odata[index_out] = idata[index_in];
}

// transpose in shared memory

__global__ void transposeShared(float *odata, float *idata, int width, int height)
{
  __shared__ float tile[NUM_THREADS][NUM_THREADS];

  int xIndex = blockIdx.x * NUM_THREADS + threadIdx.x;
  int yIndex = blockIdx.y * NUM_THREADS + threadIdx.y;  

  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * NUM_THREADS + threadIdx.x;
  yIndex = blockIdx.x * NUM_THREADS + threadIdx.y;

  int index_out = xIndex + (yIndex)*height;

  tile[threadIdx.y][threadIdx.x] = idata[index_in];

  __syncthreads();

  odata[index_out] = tile[threadIdx.x][threadIdx.y];
}

// transpose in shared memory with no bank conflicts

__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height)
{
  __shared__ float tile[NUM_THREADS][NUM_THREADS+1];

  int xIndex = blockIdx.x * NUM_THREADS + threadIdx.x;
  int yIndex = blockIdx.y * NUM_THREADS + threadIdx.y;  

  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * NUM_THREADS + threadIdx.x;
  yIndex = blockIdx.x * NUM_THREADS + threadIdx.y;

  int index_out = xIndex + (yIndex)*height;

  tile[threadIdx.y][threadIdx.x] = idata[index_in];

  __syncthreads();

  odata[index_out] = tile[threadIdx.x][threadIdx.y];
}


// transpose host

void transposeHost(float* gold, float* idata, const  int m, const  int n)
{
  for(  int y = 0; y < n; ++y) {
    for(  int x = 0; x < m; ++x) {
      gold[(x * n) + y] = idata[(y * m) + x];
    }
  }
}

int main( int argc, char** argv) 
{
  int m = SIZE;
  int n = SIZE;

  if (m%NUM_THREADS != 0 || n%NUM_THREADS != 0) {
      printf("\nMatrix size must be integral multiple of tile size\nExiting...\n\n");
      printf("FAILED\n\n");
      return 1;
  }

  const  int mem_size = sizeof(float) * m*n;

  // allocate host memory
  float *h_idata = (float*) malloc(mem_size);
  float *h_odata = (float*) malloc(mem_size);
  float *transposeGold = (float *) malloc(mem_size);  

  // allocate device memory
  float *d_idata, *d_odata;
  cudaMalloc( (void**) &d_idata, mem_size);
  cudaMalloc( (void**) &d_odata, mem_size);

  // initalize host data
  for(  int i = 0; i < (m*n); ++i)
    h_idata[i] = (float) i;
  
  // copy host data to device
  cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);

  // Compute reference transpose solution
  transposeHost(transposeGold, h_idata, m, n);

  // print out common data for all kernels
  printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n", 
	 m, n, m/NUM_THREADS, n/NUM_THREADS, NUM_THREADS, NUM_THREADS, NUM_THREADS, NUM_THREADS);
    printf ("mem_size %d \n", mem_size);

  // execution configuration parameters
  dim3 threads(NUM_THREADS, NUM_THREADS);
  dim3 grid(m/NUM_THREADS, n/NUM_THREADS);

  // timing
  float etime, bw;

  // CUDA events
  cudaEvent_t start, stop;

  // initialize events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaError_t err_cuda;
  err_cuda = cudaGetLastError()  ;


  // copy kernel

  cudaEventRecord(start);
  for (int i=0; i < NUM_REPS; i++) {
      copy<<<grid, threads>>>(d_odata, d_idata, m, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  err_cuda = cudaGetLastError() ;

  if(err_cuda != cudaSuccess)  {
      fprintf(stderr,"%s\n",cudaGetErrorString(err_cuda)) ;
  }

  cudaEventElapsedTime(&etime, start, stop);

  bw = 2.0f * 1000.0f * mem_size/(1024*1024*1024)/(etime/NUM_REPS);

  printf("%-20s time[ms] %.10e BW[GB/s] %.5f\n", "copy kernel", etime/NUM_REPS, bw);


  // transpose naive

  cudaEventRecord(start);
  for (int i=0; i < NUM_REPS; i++) {
      transposeNaive<<<grid, threads>>>(d_odata, d_idata, m, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  err_cuda = cudaGetLastError() ;

  if(err_cuda != cudaSuccess)  {
      fprintf(stderr,"%s\n",cudaGetErrorString(err_cuda)) ;
  }

  cudaEventElapsedTime(&etime, start, stop);

  bw = 2.0f * 1000.0f * mem_size/(1024*1024*1024)/(etime/NUM_REPS);

  printf("%-20s time[ms] %.10e BW[GB/s] %.5f\n", "transpose naive", etime/NUM_REPS, bw);

  cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);


  // transpose shared memory

  cudaEventRecord(start);
  for (int i=0; i < NUM_REPS; i++) {
      transposeShared<<<grid, threads>>>(d_odata, d_idata, m, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  err_cuda = cudaGetLastError() ;

  if(err_cuda != cudaSuccess)  {
      fprintf(stderr,"%s\n",cudaGetErrorString(err_cuda)) ;
  }

  cudaEventElapsedTime(&etime, start, stop);

  bw = 2.0f * 1000.0f * mem_size/(1024*1024*1024)/(etime/NUM_REPS);

  printf("%-20s time[ms] %.10e BW[GB/s] %.5f\n", "transpose shared", etime/NUM_REPS, bw);

  cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);


  // transpose shared memory no bank conflicts

  cudaEventRecord(start);
  for (int i=0; i < NUM_REPS; i++) {
      transposeNoBankConflicts<<<grid, threads>>>(d_odata, d_idata, m, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  err_cuda = cudaGetLastError() ;

  if(err_cuda != cudaSuccess)  {
      fprintf(stderr,"%s\n",cudaGetErrorString(err_cuda)) ;
  }

  cudaEventElapsedTime(&etime, start, stop);

  bw = 2.0f * 1000.0f * mem_size/(1024*1024*1024)/(etime/NUM_REPS);

  printf("%-20s time[ms] %.10e BW[GB/s] %.5f\n", "SM no bank conflicts", etime/NUM_REPS, bw);

  cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);

  // cleanup
  free(h_idata);
  free(h_odata);
  free(transposeGold);
  cudaFree(d_idata);
  cudaFree(d_odata);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  
  return 0;

}
