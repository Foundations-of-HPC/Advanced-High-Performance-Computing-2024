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



//
// Compare two matrices ... return the sum of the squares
// of the differences of the two input matrices.
//

#include "mm.h"
#include <stdio.h>
#include <omp.h>

TYPE errsqr(int Ndim, int Mdim, TYPE *C, TYPE *Cref) {
  int i, j;
  TYPE tmp, errsqr;
  errsqr = (TYPE)0.0;
  for (i = 0; i < Ndim; i++) {
    for (j = 0; j < Mdim; j++) {
      tmp = *(C + i * Mdim + j) - (*(Cref + i * Mdim + j));
      errsqr += tmp * tmp;
    }
  }
  return errsqr;
}


/*
Zero a matrix
*/
void mm_zero(TYPE *MC, int Ndim, int Mdim) {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < Ndim; i++)
    for (int j = 0; j < Mdim; j++)
      MC[i * Mdim + j] = (TYPE)0.0;
}

void mm_rand(TYPE *M, int Ndim, int Mdim) {
  #pragma omp parallel 
  {
  unsigned int seed = omp_get_thread_num();
  #pragma omp for collapse(2)
  for (int i = 0; i < Ndim; i++)
    for (int j = 0; j < Mdim; j++)
      M[i * Mdim + j] = rand_r(&seed) % 32 / 32.0f;
  }
}

/*
*   Print a Matrix
*/
void mm_print(TYPE *MC,int Ndim, int Mdim ) {
  int i, j;
  for (i = 0; i < Ndim; i++) {
    for (j = 0; j < Mdim; j++)
      printf("[%04d][%04d] = %g   ", i, j, *(MC + i * Mdim + j));
    printf("\n");
  }
}

/*
*   Initialize a matrix to CONSTANT VALUES or Progressive or zeros
*/
void mm_init(TYPE *MC, int Ndim, int Mdim, TYPE val) {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < Ndim; i++)
    for (int k = 0; k < Mdim; k++)
    #if defined (MCONST)
      MC[i * Mdim + k] = val;
    #elif defined (MPROG)
      MC[i * Mdim + k] = i * Mdim + k;
   #else
      MC[i * Mdim + k] = 0.;
   #endif
}


/*
 * DMM function.
 *
 * C = alpha * A * B
 *
 *
 * Arguments ==========
 *
 * M      - INTEGER. On entry,  M  specifies  the number  of rows  of the  matrix A,  of the  matrix  C and of
 * the  matrix  D. M  must  be at least  2.
 *
 *
 * N      - INTEGER. On entry,  N  specifies the number  of columns of the matrix B  and the number of columns
 * of the matrix C. N must be at least 2.
 *
 *
 * K      - INTEGER. On entry,  K  specifies  the number of columns of the matrix A  and the number of rows of
 * the matrix B. K must be at least  2.
 *
 *
 * ALPHA  - DOUBLE PRECISION. On entry, ALPHA specifies the scalar alpha. Unchanged on exit.
 *
 * A      - DOUBLE PRECISION array of DIMENSION ( M, K )
 *
 *
 * B      - DOUBLE PRECISION array of DIMENSION ( K, N )
 *
 * C      - DOUBLE PRECISION array of DIMENSION ( M, N ).
 *
 */


void __attribute__ ((noinline)) mm_mul_gpu(TYPE *MA, TYPE *MB, TYPE *MC, TYPE alpha, int Ndim, int Mdim, int Kdim){
    #pragma omp target teams distribute parallel for collapse(2) map(to: alpha) map(from: MC[0:Ndim*Mdim]) 
    for (int i = 0; i < Mdim; i++) {
      for (int j = 0; j < Ndim; j++) {
        for (int kk = 0; kk < Kdim; kk++) {
          MC[i * Ndim + j] =  MA [i * Kdim + kk] * MB [kk * Ndim + j] + MC[i * Ndim + j];
        }
      MC[i * Ndim + j] *= alpha ;
    }
  }
}

void __attribute__ ((noinline)) mm_mul_gpu2(TYPE *MA, TYPE *MB, TYPE *MC, TYPE alpha, int Ndim, int Mdim, int Kdim){
    #pragma omp target teams distribute parallel for simd map(to: alpha) map(from: MC[0:Ndim*Mdim]) 
    for (int i = 0; i < Mdim; i++) {
      for (int j = 0; j < Ndim; j++) {
        for (int kk = 0; kk < Kdim; kk++) {
          MC[i * Ndim + j] =  MA [i * Kdim + kk] * MB [kk * Ndim + j] + MC[i * Ndim + j];
        }
      MC[i * Ndim + j] *= alpha ;
    }
  }
}

int __attribute__((noinline)) dgemm_cpu(int *M, int *N, int *K, double *alpha,
				    double *MA, double *MB, double *beta, double *MC, double *MD)
{

#pragma omp parallel for shared(MA,MB,MC,MD)  schedule (static, 10)
	for (int i = 0; i < *M; i++) {
		for (int j = 0; j < *N; j++) {
			for (int kk = 0; kk < *K; kk++) {
				*(MD + i * *N + j) += *(MA + i * *K + kk) * *(MB + kk * *N + j);
			}
			*(MD + i * *N + j) = *alpha * *(MD + i * *N + j) + *beta * *(MC + i * *N + j);
		}
	}
	return (0);
}
