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


/*
 * - <<<2^0 , 2^0 >>>, TOO SLOW! not tested
 */
void GPUsaxpy(const int n,
            const float a,
            const float *x,
                  float *y,
            const int ial, const int itr) 
{
int m = (n >> 4);
#pragma omp target data device(0) \
  map(to:a, m, x[0:n]) map(tofrom:y[0:n])
{
  
#pragma omp target teams device(0)  num_teams(ial) \
  map(to:a, m, x[0:n]) map(tofrom:y[0:n]) \
  default(none) shared(a, m, x, y)shared(itr)
#pragma omp distribute parallel for simd num_threads(itr)\
    dist_schedule(static, itr) \
  default(none) shared(a, m, x, y) shared(x) 
for (int i = 0; i < m; ++i) {
  y[i          ] = a * x[i          ] + y[i          ];
  y[i +       m] = a * x[i +       m] + y[i +       m];
  y[i + 0x2 * m] = a * x[i + 0x2 * m] + y[i + 0x2 * m];
  y[i + 0x3 * m] = a * x[i + 0x3 * m] + y[i + 0x3 * m];
  y[i + 0x4 * m] = a * x[i + 0x4 * m] + y[i + 0x4 * m];
  y[i + 0x5 * m] = a * x[i + 0x5 * m] + y[i + 0x5 * m];
  y[i + 0x6 * m] = a * x[i + 0x6 * m] + y[i + 0x6 * m];
  y[i + 0x7 * m] = a * x[i + 0x7 * m] + y[i + 0x7 * m];
  y[i + 0x8 * m] = a * x[i + 0x8 * m] + y[i + 0x8 * m];
  y[i + 0x9 * m] = a * x[i + 0x9 * m] + y[i + 0x9 * m];
  y[i + 0xa * m] = a * x[i + 0xa * m] + y[i + 0xa * m];
  y[i + 0xb * m] = a * x[i + 0xb * m] + y[i + 0xb * m];
  y[i + 0xc * m] = a * x[i + 0xc * m] + y[i + 0xc * m];
  y[i + 0xd * m] = a * x[i + 0xd * m] + y[i + 0xd * m];
  y[i + 0xe * m] = a * x[i + 0xe * m] + y[i + 0xe * m];
  y[i + 0xf * m] = a * x[i + 0xf * m] + y[i + 0xf * m];
}
}
  }

void GPU1saxpy(const int n,
            const float a,
            const float *x,
                  float *y,
            const int ial, const int itr) 
{
int m = (n >> 4);
#pragma omp target data device(0) \
  map(to:a, m, x[0:n]) map(tofrom:y[0:n])
{
#pragma omp target teams device(0)   \
  map(to:a, m, x[0:n]) map(tofrom:y[0:n]) \
  default(none) shared(a, m, x, y) shared(itr)
#pragma omp distribute parallel for  simd  schedule(static, 32)\
  default(none) shared(a, m, x, y) shared(x) 
for (int i = 0; i < m; ++i) {
  y[i          ] = a * x[i          ] + y[i          ];
  y[i +       m] = a * x[i +       m] + y[i +       m];
  y[i + 0x2 * m] = a * x[i + 0x2 * m] + y[i + 0x2 * m];
  y[i + 0x3 * m] = a * x[i + 0x3 * m] + y[i + 0x3 * m];
  y[i + 0x4 * m] = a * x[i + 0x4 * m] + y[i + 0x4 * m];
  y[i + 0x5 * m] = a * x[i + 0x5 * m] + y[i + 0x5 * m];
  y[i + 0x6 * m] = a * x[i + 0x6 * m] + y[i + 0x6 * m];
  y[i + 0x7 * m] = a * x[i + 0x7 * m] + y[i + 0x7 * m];
  y[i + 0x8 * m] = a * x[i + 0x8 * m] + y[i + 0x8 * m];
  y[i + 0x9 * m] = a * x[i + 0x9 * m] + y[i + 0x9 * m];
  y[i + 0xa * m] = a * x[i + 0xa * m] + y[i + 0xa * m];
  y[i + 0xb * m] = a * x[i + 0xb * m] + y[i + 0xb * m];
  y[i + 0xc * m] = a * x[i + 0xc * m] + y[i + 0xc * m];
  y[i + 0xd * m] = a * x[i + 0xd * m] + y[i + 0xd * m];
  y[i + 0xe * m] = a * x[i + 0xe * m] + y[i + 0xe * m];
  y[i + 0xf * m] = a * x[i + 0xf * m] + y[i + 0xf * m];
}
}
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

   /*
   * check the number of accelerators
   */
   /*
   * check the number of accelerators
   */
  if (0 == omp_get_num_devices()) {
    printf("No accelerator found ... exit\n");
    exit(EXIT_FAILURE);
  } else {
    printf("Available devices:  %d\n",omp_get_num_devices());
    int nteams= omp_get_num_teams();
    int nthreads= omp_get_max_threads();
    printf("Running on device with %d teams in total and %d threads in each team\n",nteams,nthreads);

  }

  
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

    // Parallel initialization of x and y using rand_r
    printf("Start: Initialization Kernel...");
    #pragma omp parallel shared(x, y, n)
    {
        unsigned int seed = omp_get_thread_num(); // Initialize thread-specific seed
        #pragma omp for
        for (i = 0; i < n; ++i) {
            x[i] = rand_r(&seed) % 32 / 32.0f;
            y[i] = rand_r(&seed) % 32 / 32.0f;
        }
    }

    printf("Start: Computing Kernel with 65536 team ad 128 threads");
    tb = omp_get_wtime();
    GPUsaxpy(n, a, x, y, 0,0) ;
    te = omp_get_wtime();
    t = te - tb;
    printf("    Done\n");
    printf("Time of kernel: %lf\n", t);   
    printf("Start: Computing Kernel");
    tb = omp_get_wtime();
    GPU1saxpy(n, a, x, y, 0,0) ;
    te = omp_get_wtime();
    t = te - tb;
    printf("    Done\n");
    printf("Time of kernel: %lf\n", t);   

    free(x);
    free(y);
    free(yhost);

    exit(EXIT_SUCCESS);
}
