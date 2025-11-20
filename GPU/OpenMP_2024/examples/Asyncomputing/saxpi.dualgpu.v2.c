/*
 * Created on Fri Dec 13 2024
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



/*
Threads enabled async computing on dual GPU. 
NO STREAMS created
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define TWO26 (1 << 26) // Define TWO26 as 2^26

#pragma omp declare target 
int computeLoop(const int n,
                const float a,
                const float *x,
                      float *y,
                const int device)
{
    #pragma omp target data device(device) map(to:a, n, x[0:n]) map(tofrom:y[0:n])
    {
        #pragma omp target teams distribute parallel for simd device(device)
        for (int i = 0; i < n; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }
    return device;
}
#pragma omp end declare target

void GPUsaxpy(const int n,
              const float a,
              const float *x,
                    float *y)
{
    int num_devices = omp_get_num_devices();
    if (num_devices < 2) {
        printf("Error: At least two devices are required.\n");
        exit(EXIT_FAILURE);
    }

    int nhalf = n / 2; // Split workload into two halves

    // Use OpenMP to launch computations on two GPUs concurrently
    #pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num(); // 0 or 1
        int device = thread_id; // Assign GPU device based on thread ID

        // Calculate offsets for each GPU
        const int offset = thread_id * nhalf;
        const float *x_local = x + offset;
        float *y_local = y + offset;

        // Each GPU works on its portion of the data
        computeLoop(nhalf, a, x_local, y_local, device);
    }
}

int main(int argc, char *argv[])
{
    int i, n, iret = 0;
    size_t nbytes;
    float a = 2.0f;
    float *x, *y;
    double t = 0.0;
    double tb, te;

    n = TWO26;
    nbytes = sizeof(float) * n;
    printf("Running on n: %d elements\n", n);

    // Check the number of accelerators
    if (omp_get_num_devices() < 2) {
        printf("At least two accelerators are required.\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Available devices: %d\n", omp_get_num_devices());
    }

    // Memory allocation
    if (NULL == (x = (float *)malloc(nbytes))) iret = -1;
    if (NULL == (y = (float *)malloc(nbytes))) iret = -1;
    if (0 != iret) {
        printf("Error: Memory allocation failed.\n");
        free(x);
        free(y);
        exit(EXIT_FAILURE);
    }

    printf("Start: Initialization Kernel\n");

    // Parallel initialization of x and y
    #pragma omp parallel shared(x, y, n)
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for
        for (i = 0; i < n; ++i) {
            x[i] = rand_r(&seed) % 32 / 32.0f;
            y[i] = rand_r(&seed) % 32 / 32.0f;
        }
    }

    printf("Start: Computing Kernel\n");
    tb = omp_get_wtime();

    // Call GPUsaxpy to use both GPUs
    GPUsaxpy(n, a, x, y);

    te = omp_get_wtime();
    t = te - tb;

    printf("Time of kernel: %lf seconds\n", t);

    free(x);
    free(y);

    exit(EXIT_SUCCESS);
}