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

void async_device_task(int *data, int N, omp_event_t event) {
    // Simulating an asynchronous task (e.g., device computation)
    printf("Starting device task...\n");
    #pragma omp target map(tofrom: data[0:N])
    {
        for (int i = 0; i < N; i++) {
            data[i] += 1; // Simple computation
        }
    }

    // Fulfill the event to signal task completion
    omp_fulfill_event(event);
    printf("Device task completed.\n");
}

int main() {
    const int N = 10;
    int *data = (int *)malloc(sizeof(int) * N);

    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = i;
    }

    // Create an OpenMP event
    omp_event_t event;
    omp_init_event(&event);

    // Launch a detached task
    #pragma omp task detach(event)
    {
        async_device_task(data, N, event);
    }

    // Perform other computations while the task is detached
    printf("Host is performing other work...\n");

    // Wait for all tasks to complete
    #pragma omp taskwait

    // Clean up the event
    omp_destroy_event(&event);

    // Check results
    printf("Final data:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", data[i]);
    }
    printf("\n");

    free(data);
    return 0;
}