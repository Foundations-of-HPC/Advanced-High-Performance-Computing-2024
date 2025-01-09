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

#define ROWS (1 << 13) /* 4GB*/
#define COLS (1 << 13)
#define NUM_CALC 8

// Funzione generica per allocare e inizializzare una matrice o un vettore
float* allocate_and_initialize_array(size_t rows, size_t cols) {
    size_t total_elements = rows * cols;
    float* array = (float*)malloc(total_elements * sizeof(float));
    if (!array) {
        fprintf(stderr, "Errore: allocazione memoria fallita!\n");
        exit(EXIT_FAILURE);
    }

    // Inizializzazione casuale
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() + time(NULL);
        #pragma omp for
        for (size_t i = 0; i < total_elements; i++) {
            array[i] = (float)(rand_r(&seed) % 100) / 10.0f; // Numeri casuali tra 0 e 10
        }
    }
    #pragma omp target enter data map(to:array[:total_elements])
    return array;
}

void deallocate (float* array, size_t n)
{
    #pragma omp target exit data map(delete:array[:n])
    free(array);
}

// Funzione GEMV: Calcola y = alpha * A * x + beta * y
void gemv(float alpha, float* matrix, float* vector, float beta, float* result, size_t rows, size_t cols) {
    #pragma omp target teams distribute  map(to: matrix[0:rows*cols], vector[0:cols], alpha, beta) map(from: result[0:rows])
    for (size_t i = 0; i < rows; i++) {
        float temp = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            temp += matrix[i * cols + j] * vector[j];
        }
        result[i] = alpha * temp + beta * result[i];
    }
}


int main() {
    unsigned long dim; 
    double t = 0.0;
    double tb, te;
    dim = (long long)ROWS*(long long)COLS;
    double size_in_gib = (double)(dim* sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    // Allocazione e inizializzazione della matrice e dei vettori
    printf("matrice of size %d,%d of total size %f Gb\n", ROWS, COLS, size_in_gib);

    float* manyA[NUM_CALC];
    float* manyX[NUM_CALC];
    float* manyY[NUM_CALC];

    for (int i = 0; i < NUM_CALC; i++) {
        manyA[i] = allocate_and_initialize_array(ROWS, COLS);
        manyX[i] = allocate_and_initialize_array(COLS, 1);
        manyY[i] = allocate_and_initialize_array(ROWS, 1);
    }



    // Valori di alpha e beta
    float alpha = 1.0f, beta = 0.0f;

    // Calcolo della GEMV
    tb = omp_get_wtime();
    for (int i = 0; i < NUM_CALC; i++) {
        gemv(alpha, manyA[i], manyX[i], beta, manyY[i], ROWS, COLS);
        float* __restrict__ Yout = manyY[i];
        #pragma omp target update from(Yout[:ROWS])
    }
    te = omp_get_wtime();
    t = te - tb;
    printf("Time of GEMV: %lf\n", t);

    // Deallocazione
     for (int i = 0; i < NUM_CALC; i++) {
        deallocate(manyA[i], ROWS*COLS);
        deallocate(manyX[i], COLS);
        deallocate(manyY[i], ROWS);
    }
    
    return 0;
}