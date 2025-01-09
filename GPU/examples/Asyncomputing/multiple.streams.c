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

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void gpu_task(int *data, int start, int end) {
    #pragma omp target map(tofrom: data[start:end])
    {
        for (int i = start; i < end; i++) {
            data[i] += 1; // Simple GPU computation
        }
    }
}

int main() {
    const int N = 1000;
    const int num_streams = 4;
    int *data = (int *)malloc(sizeof(int) * N);

    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = i;
    }

    // Partition data into chunks for each "stream"
    int chunk_size = N / num_streams;

    printf("Creating %d streams using OpenMP tasks...\n", num_streams);

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int stream = 0; stream < num_streams; stream++) {
                int start = stream * chunk_size;
                int end = (stream == num_streams - 1) ? N : start + chunk_size;

                // Asynchronous task for each stream
                #pragma omp task firstprivate(start, end) nowait
                gpu_task(data, start, end);
            }
        }

        // Wait for all tasks to complete
        #pragma omp taskwait
    }

    printf("All streams completed.\n");

    // Verify results
    for (int i = 0; i < N; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    free(data);
    return 0;
}