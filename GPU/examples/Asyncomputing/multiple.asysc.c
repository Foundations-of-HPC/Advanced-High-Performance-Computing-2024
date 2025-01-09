/*
 * Created on Wed Dec 18 2024
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
 * Copyright (c) 2025 Taffoni
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

#define SIZE 4

// Function to initialize the matrix with thread-safe random numbers
void initialize_matrix_threadsafe(int matrix[SIZE][SIZE]) {
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() + time(NULL);
        #pragma omp for collapse(2)
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                matrix[i][j] = rand_r(&seed) % 10; // Random values [0-9]
            }
        }
    }
}

// Function to print the matrix
void print_matrix(const char* name, int matrix[SIZE][SIZE]) {
    printf("%s:\n", name);
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Matrix multiplication on the GPU
void gpu_matrix_multiply(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    #pragma omp target teams distribute parallel for nowait collapse(2) map(to: A[0:SIZE][0:SIZE], B[0:SIZE][0:SIZE]) map(from: C[0:SIZE][0:SIZE])
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            C[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // Allocate and initialize matrices
    int A[SIZE][SIZE], B[SIZE][SIZE], X[SIZE][SIZE], Y[SIZE][SIZE];
    int C[SIZE][SIZE] = {0}, D[SIZE][SIZE] = {0};
    int E[SIZE][SIZE], F[SIZE][SIZE] = {0}, CF[SIZE][SIZE] = {0};

    initialize_matrix_threadsafe(A);
    initialize_matrix_threadsafe(B);
    initialize_matrix_threadsafe(X);
    initialize_matrix_threadsafe(Y);
    initialize_matrix_threadsafe(E);

   // printf("Matrix Operations with OpenMP Offloading to GPU\n");
   // print_matrix("Matrix A", A);
   // print_matrix("Matrix B", B);
   // print_matrix("Matrix X", X);
   // print_matrix("Matrix Y", Y);
   // print_matrix("Matrix E", E);

    // Perform matrix multiplications on the GPU
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Compute A x B -> C
            #pragma omp task  depend(out: C) 
            gpu_matrix_multiply(A, B, C);

            // Compute X x Y -> D
            #pragma omp task  depend(out: D) 
            gpu_matrix_multiply(X, Y, D);

            // Wait for C and D to be ready before proceeding
            #pragma omp taskwait

            // Compute D x E -> F
            #pragma omp task  depend(in: D, E) depend(out: F) 
            gpu_matrix_multiply(D, E, F);

            // Compute C x F -> CF
            #pragma omp task  depend(in: C, F) depend(out: CF) 
            gpu_matrix_multiply(C, F, CF);

            // Ensure all tasks are complete before ending
            #pragma omp taskwait
        }
    } // End of parallel region

    // Print the resulting matrices
    //print_matrix("Matrix C", C);
    //print_matrix("Matrix D", D);
    //print_matrix("Matrix F", F);
    //print_matrix("Matrix C x F (Result)", CF);

    return 0;
}