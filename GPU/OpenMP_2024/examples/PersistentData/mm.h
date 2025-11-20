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

#define TCPU_TIME (clock_gettime( id, &ts ), (double)ts.tv_sec + \
																			(double)ts.tv_nsec * 1e-9)

#define TYPE double
#if !defined (MCONST) && !defined (MPROG)
#define MCONST
#endif

/*
* CPU gpu defintions
*/
#define STRINGIFY(a) #a
#define ACC_APPLY_PRAGMA(...)  _Pragma(STRINGIFY(__VA_ARGS__))
#if defined (GPU)
	#define ACC_PRAGMA(...) ACC_APPLY_PRAGMA( omp __VA_ARGS__)
	#define ACC_FUNCTION_BEGIN(...) _Pragma("omp declare target")
	#define ACC_FUNCTION_END(...) _Pragma("omp end declare target")
#elif defined (CPU)
	#define ACC_PRAGMA(...) ACC_APPLY_PRAGMA( omp __VA_ARGS__)
	#define ACC_FUNCTION_BEGIN(...) _Pragma("omp declare target")
	#define ACC_FUNCTION_END(...) _Pragma("omp end declare target")
#else
	#define ACC_PRAGMA(...)
	#define ACC_FUNCTION_BEGIN(...)
	#define ACC_FUNCTION_END(...)
#endif



double errsqr(int Ndim, int Mdim, TYPE *C, TYPE *Cref);
void mm_zero(TYPE *C, int Ndim, int Mdim );
void mm_print(TYPE *C, int Ndim, int Mdim );
void mm_init(TYPE *A, int Ndim, int Mdim, TYPE val);

void __attribute__ ((noinline)) mm_mul_gpu(TYPE *A, TYPE *B, TYPE *C, TYPE alpha, int Ndim, int Mdim, int Kdim);
void __attribute__ ((noinline)) mm_mul_gpu2(TYPE *A, TYPE *B, TYPE *C, TYPE alpha, int Ndim, int Mdim, int Kdim);


void __attribute__ ((noinline)) mm_mul_cpu(TYPE *A, TYPE *B, TYPE *C, TYPE alpha, int Ndim, int Mdim, int Kdim);

