#include <stdio.h>
#include <assert.h>
#define epsilon (float)1e-5
#define DATA float
#define THREADxBLOCKalongXorY 16

void MatrixMulOnHost(DATA* M, DATA* N, DATA* P, int Width) {
  for (int i = 0; i < Width; ++i) {
    for (int j = 0; j < Width; ++j) {
      DATA pvalue = 0;
      for (int k = 0; k < Width; ++k) {
        DATA a = M[i * Width + k];
        DATA b = N[k * Width + j];
        pvalue += a * b;
      }
      P[i * Width + j] = pvalue;
    }
  }
}

__global__ void MatrixMulKernel(DATA* dM, DATA* dN, DATA* dP, int Width) {

   // INSERT CODE: index definitions
   int j = 
   int i = 

   // INSERT CODE: fastest index to threads along x direction
      DATA pvalue = 0;
      for (int k = 0; k < Width; ++k) {
        DATA a = 
        DATA b = 
        pvalue += 
      }
      dP[ ] = 
}

void MatrixMulOnDevice(DATA* M, DATA* N, DATA* P, int Width) {

  int size = Width * Width * sizeof(DATA);
  float mflops;
  DATA *dM, *dN, *dP;

  // CUDA grid management
  int gridsize = Width/THREADxBLOCKalongXorY;
  if(gridsize*THREADxBLOCKalongXorY<Width) {
    gridsize=gridsize+1;
  }
  dim3 dimGrid(gridsize,gridsize);
  dim3 dimBlock(THREADxBLOCKalongXorY, THREADxBLOCKalongXorY);
  printf("Gridsize: %d\n", gridsize);

  cudaMalloc(&dM, size);
  cudaMemcpy(dM, M, size, cudaMemcpyHostToDevice);
  cudaMalloc(&dN, size);
  cudaMemcpy(dN, N, size, cudaMemcpyHostToDevice);
  cudaMalloc(&dP, size);

  // cudaGetLastError call to reset previous CUDA errors
  // INSERT CODE
  cudaError_t mycudaerror  ;

  // Create start and stop CUDA events 
  // INSERT CODE
  cudaEvent_t start, stop;

  // kernel launch
  MatrixMulKernel<<<dimGrid, dimBlock>>>(dM, dN, dP, Width);

  // device synchronization and cudaGetLastError call
  // INSERT CODE

  // event record, synchronization, elapsed time and destruction
  // INSERT CODE
  float elapsed;
  elapsed = elapsed/1000.f; // convert to seconds

  // calculate Mflops
  // INSERT CODE
  printf("Kernel elapsed time %fs \n", elapsed);
  printf("Mflops: %f\n", mflops);

  // copy back results from device
  cudaMemcpy(P, dP, size, cudaMemcpyDeviceToHost);

  // free memory on device
  cudaFree(dM);
  cudaFree(dN);
  cudaFree(dP);

}

// main
int main(int argc, char** argv) {

  int Width;

  DATA *M, *N, *hP, *gP;

  if(argc<2) {
    fprintf(stderr,"Usage: %s Width\n",argv[0]);
    exit(1);
  }

  Width=atoi(argv[1]);

  if(Width<1) {
    fprintf(stderr,"Error Width=%d, must be > 0\n",Width);
    exit(1);

  }

  M=(DATA *)malloc(Width*Width*sizeof(DATA));
  N=(DATA *)malloc(Width*Width*sizeof(DATA));
  hP=(DATA *)malloc(Width*Width*sizeof(DATA));
  gP=(DATA *)malloc(Width*Width*sizeof(DATA));

  if(M==NULL) {
    fprintf(stderr,"Could not get memory for M\n");
    exit(1);
  }
  if(N==NULL) {
    fprintf(stderr,"Could not get memory for N\n");
    exit(1);
  }
  if(hP==NULL) {
    fprintf(stderr,"Could not get memory for hP\n");
    exit(1);
  }
  if(gP==NULL) {
    fprintf(stderr,"Could not get memory for gP\n");
    exit(1);
  }

  memset(gP,0,Width*Width*sizeof(DATA));
  memset(hP,0,Width*Width*sizeof(DATA));

  for(int y=0; y<Width; y++){
    for(int x=0; x<Width; x++) {
      M[y*Width+x]=(DATA)(((y+1)*Width+x+1)/(Width));
      N[y*Width+x]=(DATA)(((y+1)*Width+x+1)/(Width));
    }
  }

  MatrixMulOnHost(M, N, hP, Width);

  MatrixMulOnDevice(M, N, gP, Width);

  int errCnt = 0;
  for(int y=0; y<Width; y++){
    for(int x=0; x<Width; x++) {
      DATA it = hP[y*Width+x];
      if(fabs(it - gP[y*Width+x])> epsilon*it) {
        printf("failing x=%d, y=%d: %f!=%f \n",x,y,it,gP[y*Width+x]);
        errCnt++;
      }
    }
  }

  if(errCnt==0)
    printf("\nTEST PASSED\n");
  else
    printf("\n\nTEST FAILED: number of errors:  %d\n", errCnt);

}
