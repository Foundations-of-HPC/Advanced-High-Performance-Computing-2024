#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM 256

#define REPEAT 1


__global__ void strideCopy(float *odata, float* idata, int stride)
{
    int xid = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    odata[xid] = idata[xid];
}

__global__ void offsetCopy(float *odata, float* idata, int offset)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
    odata[xid] = idata[xid];
}

void init_data(float * buffer, int size) {
   for ( int i = 0 ; i < size ; i++ ) {
      buffer[i] = (float) i;
   }
}

int main (int argc, char *argv[]) {
  
   int buffer_size;

   float * h_buffer;

   float * d_inp;
   float * d_out;

   cudaEvent_t start, end;
   float eventEtime, eventBandwidth;

   int size = 65535;
   size *= BLOCK_DIM;

   int stride = 1;
   int offset = 0;

   if (argc > 1) {
      stride = atoi(argv[1]);
   }

   if (argc > 2) {
      offset = atoi(argv[2]);
   }

   buffer_size = (size+offset)*stride;

   dim3 block (BLOCK_DIM);
   dim3 grid  ( (size-1)/block.x+1 );

   printf("Number of Elemets = %d\n", size);
   printf("Stride            = %d\n", stride);
   printf("Offset            = %d\n", offset);
   printf("Buffer size       = %d MB\n", sizeof(float)*buffer_size/(1<<20));

   printf("Block size        = %d\n", block.x);
   printf("Grid size         = %d\n", grid.x);

   printf("Repeat            = %d\n", REPEAT);

//-- insert C code -------------
// allocate memory on host
   h_buffer = (float *) malloc (sizeof(float)*buffer_size);
//------------------------------

   init_data (h_buffer, buffer_size);

//-- insert CUDA code ----------
// creat cuda events (start, end) for timing
   cudaEventCreate(&start);
   cudaEventCreate(&end);
//------------------------------

//-- insert CUDA code ----------
// allocate memory buffers on selected GPU
   cudaMalloc((void**)&d_inp, sizeof(float)*buffer_size);
   cudaMalloc((void**)&d_out, sizeof(float)*buffer_size);
//------------------------------

   cudaMemcpy(d_inp, h_buffer, sizeof(float)*buffer_size, cudaMemcpyHostToDevice);

   printf ("\nStride Copy : Stride = %d\n", stride);

   cudaEventRecord(start);

//-- insert CUDA code ----------
// launch stride copy kernel
   for ( int i=0 ; i < REPEAT ; i++ ) {
      strideCopy <<<grid, block>>> (d_out, d_inp, stride);
   }
//------------------------------

   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&eventEtime, start, end);

   eventEtime /= REPEAT;

   eventBandwidth = (float) ( 2.0F*size*sizeof(float) ) / ( eventEtime );
   printf("Time / ms        = %.1f\n", eventEtime);
   printf("Bandwidth / GB/s = %.2f\n", eventBandwidth / 1.e6);

   printf ("\nOffset Copy : Offset = %d\n", offset);

   cudaEventRecord(start);

//-- insert CUDA code ----------
// launch offset copy kernel
   for ( int i=0 ; i < REPEAT ; i++ ) {
      offsetCopy <<<grid, block>>> (d_out, d_inp, offset);
   }
//------------------------------

   cudaEventRecord(end);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&eventEtime, start, end);

   eventEtime /= REPEAT;

   eventBandwidth = (float) ( 2.0F*size*sizeof(float) ) / ( eventEtime );
   printf("Time / ms        = %.1f\n", eventEtime);
   printf("Bandwidth / GB/s = %.2f\n", eventBandwidth / 1.e6);

   cudaMemcpy(h_buffer, d_out, sizeof(float)*buffer_size, cudaMemcpyDeviceToHost);

//-- insert CUDA code ----------
// free resources on device (memory buffers and events)
   cudaEventDestroy(start);
   cudaEventDestroy(end);

   cudaFree(d_inp);
   cudaFree(d_out);
//----------------------------

//-- insert C coda -----------
// free resources on host
   free(h_buffer);
//------------------------------

   return 0;
}

