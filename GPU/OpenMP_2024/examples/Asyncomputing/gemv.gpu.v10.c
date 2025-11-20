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

#define ROWS (1 << 12) /* 4GB*/

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
    return array;
}

void deallocate (float* array, size_t n)
{
    free(array);
}
#pragma omp declare target
void vect( const float a,
                const float *x,
                const float *y, float *yout,const int n)
{
        #pragma omp target teams distribute parallel for simd map(to:a, n, x[0:n]) map(to:y[0:n]) map(tofrom:yout[0:n])  num_teams(1) num_threads(1024) nowait
        for (int i = 0; i < n; ++i) {
            yout[i] = a * x[i] + y[i];
        }
}
#pragma omp end declare target

// Funzione GEMV: Calcola y = alpha * A * x + beta * y
#pragma omp declare target
void gemv(float alpha, float* matrix, float* vector, float beta, float* result, size_t rows, size_t cols) {
   #pragma omp target teams distribute parallel for map(to: matrix[0:rows*cols], vector[0:cols], alpha, beta) map(tofrom: result[0:rows]) num_teams(1) num_threads(1024) nowait
    for (size_t i = 0; i < rows; i++) {
        float temp = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            temp += matrix[i * cols + j] * vector[j];
        }
        result[i] = alpha * temp + beta * result[i];
    }
}
#pragma omp end declare target


int main() {
    unsigned long dim; 
    double t = 0.0;
    double tb, te;
    dim = (long long)ROWS*(long long)ROWS;
    double size_in_gib = (double)(dim* sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    // Allocazione e inizializzazione della matrice e dei vettori
    printf("matrice of size %d,%d of total size %f Gb\n", ROWS, ROWS, size_in_gib);

    float *MA, *MB, *MC;
    float *X1, *X2, *X3;
    float *Y1, *Y2, *Y3, *Y4, *YOUT;
   
    // Valori di alpha e beta
    float alpha = 1.0f, beta = 0.0f;

    // Calcolo della GEMV
    tb = omp_get_wtime();  
 
    MA = allocate_and_initialize_array(ROWS, ROWS);
    MB = allocate_and_initialize_array(ROWS, ROWS);
    MC = allocate_and_initialize_array(ROWS, ROWS);
    X1 = allocate_and_initialize_array(ROWS, 1);
    X2 = allocate_and_initialize_array(ROWS, 1);
    Y1 = allocate_and_initialize_array(ROWS, 1); 
    Y2 = allocate_and_initialize_array(ROWS, 1); 
    Y3 = allocate_and_initialize_array(ROWS, 1); 
    Y4 = allocate_and_initialize_array(ROWS, 1); 
    YOUT = allocate_and_initialize_array(ROWS, 1); 
    #pragma omp parallel
    {
    #pragma omp single
    {
    {
    #pragma omp target  map(to: MA[0:ROWS*ROWS], X1[0:ROWS], alpha, beta) map(from: Y1[0:ROWS]) \
    nowait depend(out:Y1) //depend(in:MA,X1,Y1)
    gemv(alpha, MA, X1, beta, Y1, ROWS, ROWS); // A1: parallel
    
    #pragma omp target  map(to: MB[0:ROWS*ROWS], X2[0:ROWS], alpha, beta) map(from: Y2[0:ROWS]) \
    nowait depend(out:Y2) //depend(in:MB,X2,Y2)
    gemv(alpha, MB, X2, beta, Y2, ROWS, ROWS); // A2: parallel

    #pragma omp target  map(to: MC[0:ROWS*ROWS], Y1[0:ROWS], alpha, beta) map(from: Y3[0:ROWS]) \
    nowait depend(in:Y1) depend(out:Y3) 
    gemv(alpha, MC, Y1, beta, Y3, ROWS, ROWS); // A3: depend  on A1
    
    #pragma omp target  map(to: MC[0:ROWS*ROWS], Y2[0:ROWS], alpha, beta) map(from: Y4[0:ROWS]) \
    nowait depend(in:Y2) depend(out:Y4)
    gemv(alpha, MC, Y2, beta, Y4, ROWS, ROWS); // A4: depend  on A2
    
    #pragma omp target  map(to: Y3[0:ROWS], Y4[0:ROWS], alpha, beta) map(from: YOUT[0:ROWS]) \
    nowait depend(in:Y3,Y4) depend(out:YOUT)
    vect(alpha, Y3, Y4, YOUT, ROWS); // A5: depend  on A3 and A4
    }
    }
        #pragma omp taskwait
    }
    te = omp_get_wtime();
    t = te - tb;
    printf("Time of GEMV: %lf\n", t);

    // Deallocazione
    // for (int i = 0; i < NUM_CALC; i++) {
    //    float* __restrict__ Yout = manyY[i];
    //    #pragma omp target update from(Yout[:ROWS])
        deallocate(MA, ROWS*ROWS);
        deallocate(MB, ROWS*ROWS);
        deallocate(MC, ROWS*ROWS);
        deallocate(X1, ROWS);
        deallocate(X2, ROWS);
        deallocate(Y1, ROWS);
        deallocate(Y2, ROWS);
        deallocate(Y3, ROWS);
        deallocate(Y4, ROWS);
        deallocate(YOUT, ROWS);
    //}
    
    return 0;
}