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
#include <time.h>
#include "mm.h"

int main(int argc, char *argv[]) {
    TYPE alpha = 1.0;
    int M = 5000, N = 3000, K = 700;
    TYPE val = 1.0;
	struct timespec	ts;
    clockid_t	id = CLOCK_PROCESS_CPUTIME_ID;
    TYPE *A = (TYPE *)malloc(sizeof(TYPE) * M * K);
    TYPE *B = (TYPE *)malloc(sizeof(TYPE) * K * N);
    TYPE *C = (TYPE *)malloc(sizeof(TYPE) * M * N);

    mm_init(A, M, K, val);
    mm_init(B, K, N, val);
    mm_zero(C, M, N);

    printf("Asynchronous data transfer using nowait...\n");

    double startTime = TCPU_TIME;

    // Asynchronously transfer data to the device
    #pragma omp target enter data nowait map(to:A[0:M*K], B[0:K*N], C[0:M*N])

    // Perform other tasks on the host while data is being transferred
    printf("Host is performing initialization while data transfer occurs...\n");

    // Wait for the data transfer to complete before starting computation
    #pragma omp taskwait

    // Task 1: Matrix multiplication
    #pragma omp target map(to:A[0:M*K], B[0:K*N]) map(from:C[0:M*N]) \
        nowait depend(out:C) depend(in:A) depend(in:B)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] *= alpha;
        }
    }
    // Task 2: Another operation depending on the result of Task 1
    #pragma omp target map(to:C[0:M*N]) map(from:B[0:K*N]) \
        nowait depend(in:C) depend(out:B)
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] += C[i * N + j % M]; // Some dependent operation
        }
    }

    // Wait for all tasks to complete
    #pragma omp taskwait

    double stopTime = TCPU_TIME;
    printf("Total Execution Time: %e seconds\n", stopTime - startTime);

    #pragma omp target exit data map(delete:A, B, C)

    free(A);
    free(B);
    free(C);

    return 0;
}