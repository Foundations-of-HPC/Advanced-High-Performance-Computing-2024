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
Task enabled async computing on dual GPU. 
Enable STREAMS
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
    #pragma omp target data  device(device) map(to:a, n, x[0:n]) map(tofrom:y[0:n]) 
    {
        #pragma omp target teams distribute parallel for simd device(device) nowait
        for (int i = 0; i < n; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }
    return device;
}
#pragma omp end declare target

void GPUsaxpy_tasks(const int n,
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

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int device = 0; device < 2; ++device) {
                int offset = device * nhalf;
                const float *x_local = x + offset;
                float *y_local = y + offset;

                // Create a task for each GPU
                #pragma omp task shared(a, x_local, y_local, nhalf) firstprivate(device) 
                {
                    computeLoop(nhalf, a, x_local, y_local, device);
                }
            }
        } // End of single region

        // Wait for all tasks to complete
        #pragma omp taskwait
    } // End of parallel region
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

    // Call GPUsaxpy_tasks to use both GPUs asynchronously
    GPUsaxpy_tasks(n, a, x, y);

    te = omp_get_wtime();
    t = te - tb;

    printf("Time of kernel: %lf seconds\n", t);

    free(x);
    free(y);

    exit(EXIT_SUCCESS);
}
