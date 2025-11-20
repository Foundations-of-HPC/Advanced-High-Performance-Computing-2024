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
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mm.h"


/*
 * Serial MATMUL function.
 *
 * D = alpha * A * B
 *
 * A      - DOUBLE PRECISION array of DIMENSION ( M, K )
 *
 * B      - DOUBLE PRECISION array of DIMENSION ( K, N )
 *
 * C      - DOUBLE PRECISION array of DIMENSION ( M, N ).
 *
 *
 * ALPHA  - DOUBLE PRECISION.
 *
 * NOTE: for a  matrix we have (row,column)
 */



int
main(int argc, char *argv[])
{

	TYPE	alpha = 1.0;
	int	M = 5000;
	int	N = 3000;
	int	K = 700;
	int isgpu=0;
	TYPE val = 1.;
	TYPE startTime, stopTime, wtime;
	struct timespec	ts;
	clockid_t	id = CLOCK_PROCESS_CPUTIME_ID;
	/*
	       CLOCK_REALTIME
	              System-wide clock that measures real (i.e., wall-clock) time.  Setting this clock requires appropriate privileges.  This clock is affected by discontinuous jumps in the sys‐
	              tem time (e.g., if the system administrator manually changes the clock), and by the incremental adjustments performed by adjtime(3) and NTP.
	       CLOCK_REALTIME_COARSE (since Linux 2.6.32; Linux-specific)
	              A faster but less precise version of CLOCK_REALTIME.  Use when you need very fast, but not fine-grained timestamps.
	       CLOCK_MONOTONIC
	              Clock that cannot be set and represents monotonic time since some unspecified starting point.  This clock is not affected by discontinuous jumps in the system time (e.g., if
	              the system administrator manually changes the clock), but is affected by the incremental adjustments performed by adjtime(3) and NTP.
	       CLOCK_MONOTONIC_COARSE (since Linux 2.6.32; Linux-specific)
	              A faster but less precise version of CLOCK_MONOTONIC.  Use when you need very fast, but not fine-grained timestamps.
	       CLOCK_MONOTONIC
	              Clock that cannot be set and represents monotonic time since some unspecified starting point.  This clock is not affected by discontinuous jumps in the system time (e.g., if
	              the system administrator manually changes the clock), but is affected by the incremental adjustments performed by adjtime(3) and NTP.
	       CLOCK_MONOTONIC_COARSE (since Linux 2.6.32; Linux-specific)
	              A faster but less precise version of CLOCK_MONOTONIC.  Use when you need very fast, but not fine-grained timestamps.
	       CLOCK_MONOTONIC_RAW (since Linux 2.6.28; Linux-specific)
	              Similar to CLOCK_MONOTONIC, but provides access to a raw hardware-based time that is not subject to NTP adjustments or the incremental adjustments performed by adjtime(3).
	       CLOCK_BOOTTIME (since Linux 2.6.39; Linux-specific)
	              Identical  to CLOCK_MONOTONIC, except it also includes any time that the system is suspended.  This allows applications to get a suspend-aware monotonic clock without having
	              to deal with the complications of CLOCK_REALTIME, which may have discontinuities if the time is changed using settimeofday(2).
	       CLOCK_PROCESS_CPUTIME_ID (since Linux 2.6.12)
	              Per-process CPU-time clock (measures CPU time consumed by all threads in the process).
	       CLOCK_THREAD_CPUTIME_ID (since Linux 2.6.12)
	              Thread-specific CPU-time clock.
	 */

	/*
	 * Memory allocation
	 *
	 * I want A, B, C, D to be contiguously allocated. On slaves we can allocate less memory
	 *
	 */
	int device_num = omp_get_device_num();
	printf("----------------------------------------\n");
	printf("MATMUL implementation = a * A * B       \n");
	printf("----------------------------------------\n");
    printf("Number of avilable devices: %d\n", omp_get_num_devices());
	printf("Default device:             %d\n", omp_get_default_device());
	printf("Current device:             %d\n", omp_get_place_num());
	printf("Number of Threads:          %d\n", omp_get_max_threads());


	TYPE *A  = (TYPE *)malloc(sizeof(TYPE *) * M * K);
	TYPE *B  = (TYPE *)malloc(sizeof(TYPE *) * K * N);
	TYPE *C  = (TYPE *)malloc(sizeof(TYPE *) * M * N);
    TYPE *C2 = (TYPE *)malloc(sizeof(TYPE *) * M * N);
	/*
	 * Matrix initialization A=1., B=1. and C=0.
	 */
	mm_init(A,M,K, val);
    mm_init(B,K,N,val);
    #pragma omp target nowait enter data map(to:A[0:M*K], B[0:N*K])
    mm_zero(C,M,N);
	#pragma omp target enter data map(to:C[0:M*N])

    mm_zero(C2,M,N);

#ifdef _DEBUG_
	if (N < 10) {
		mm_print(A, M, K);
		mm_print(B, K, N);
		printf("----------------------------------------\n");
		mm_print(C, M, N);
		printf("----------------------------------------\n");
		mm_print(C2,M,N);
	}
#endif


	printf("----------------------------------------\n");
	printf("Begin DGEMM on GPU.\n");
	startTime = TCPU_TIME;


	mm_mul_gpu(A, B, C,  alpha,  N,  M,  K);


	stopTime = TCPU_TIME;
	wtime = stopTime - startTime;
	printf("Execution Time on GPU collapse =  %e [sec]\n", wtime);
	startTime = TCPU_TIME;


	mm_mul_gpu2(A, B, C,  alpha,  N,  M,  K);

	stopTime = TCPU_TIME;
	wtime = stopTime - startTime;
	printf("Execution Time on GPU  =  %e [sec]\n", wtime);
	printf("----------------------------------------\n");
#pragma omp target exit data map(delete:A,B)
	printf("End Computation\n");

#ifdef _DEBUG_
    TYPE error;
		error=errsqr(N, M, C, C2);
		printf("Error in MM GPU= %15.10g\n",error);
		if (N < 10) {
		mm_print(C, M, N);
		printf("----------------------------------------\n");
		mm_print(C2, M, N);
	}
#endif
	return 0;
}
