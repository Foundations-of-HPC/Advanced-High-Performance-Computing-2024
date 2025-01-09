/*
 * Created on Thu Dec 19 2024
 *
 * This file is part of the exercises for the Lectures on
 * Foundations of High Performance Computing
 * given at
 *     Master in HPC and
 *     Master in Data Science and Scientific Computing
 *
 * @ SISSA, ICTP and University of Trieste
 *
 * contact: taffoni@oats.inaf.it
 *
 *
 *
 *
 * The MIT License (MIT)
 * Copyright (c) 2024 Taffoni
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
 * TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define TWO26 (1 << 26) // Define TWO26 as 2^26

#pragma omp declare target 
int computeLoop(const int n,
              const float a,
              const float *x,
                  float *y,
              const int device){
#pragma omp target data  device(device) map(to:a, n, x[0:n]) map(tofrom:y[0:n]) 
  {
#pragma omp target teams distribute parallel for simd device(device)  
    for (int i = 0; i < n; ++i) {
      y[i] = a * x[i] + y[i];
    }
  }
  return device;           
}
#pragma omp end declare target 

/*
 * - <<<2^0 , 2^0 >>>, TOO SLOW! not tested
 */
void GPUsaxpy(const int n,
            const float a,
            const float *x,
                  float *y,
            const int ial) 
{
  int GPUdevice=0;
  int nhalf=n/omp_get_num_devices();
  computeLoop(nhalf, a, x, y,GPUdevice);
}


int main(int argc, char *argv[])
{
    int i, n, iret = 0;
    size_t nbytes;
    float a = 2.0f;
    float *x, *y, *yhost;
    double t = 0.0;
    double tb, te;

    n = TWO26;
    nbytes = sizeof(float) * n;
    printf("Running on n: %d elements\n", n);

   /*
   * check the number of accelerators
   */
  if (0 == omp_get_num_devices()) {
    printf("No accelerator found ... exit\n");
    exit(EXIT_FAILURE);
  } else {
    printf("Available devices:  %d\n",omp_get_num_devices());

  }

  //#pragma omp target device(0)
  //{
  //  int nteams= omp_get_num_teams();
  //  int nthreads= omp_get_max_threads();
  //  printf("Running on device with %d teams in total and %d threads in each team\n",nteams,nthreads);
  //}

    if (NULL == (x = (float *)malloc(nbytes))) iret = -1;
    if (NULL == (y = (float *)malloc(nbytes))) iret = -1;
    if (NULL == (yhost = (float *)malloc(nbytes))) iret = -1;
    if (0 != iret) {
        printf("error: memory allocation\n");
        free(x);
        free(y);
        free(yhost);
        exit(EXIT_FAILURE);
    }
    printf("Start: Initialization Kernel\n");
    // Parallel initialization of x and y using rand_r
    #pragma omp parallel shared(x, y, n)
    {
        unsigned int seed = omp_get_thread_num(); // Initialize thread-specific seed
        #pragma omp for
        for (i = 0; i < n; ++i) {
            x[i] = rand_r(&seed) % 32 / 32.0f;
            y[i] = rand_r(&seed) % 32 / 32.0f;
        }
    }
    printf("Start: Computing Kernel\n");
    tb = omp_get_wtime();
    GPUsaxpy(n, a, x, y, 1) ;
    te = omp_get_wtime();
    t = te - tb;

    printf("Time of kernel: %lf\n", t);

    free(x);
    free(y);
    free(yhost);

    exit(EXIT_SUCCESS);
}
