#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "../headers/timing.h"
#include "../headers/vector_pragmas.h"


#if defined(USE_RESTRICT)
#define RESTRICT 1
double __attribute__ ((noinline)) process ( double *restrict A, double *restrict B, double *restrict C, int N )
#else
#define RESTRICT 0
double __attribute__ ((noinline)) process ( double *A, double *B, double *C, int N )
#endif
{
  double sum = 0;
  
  IVDEP
  LOOP_VECTORIZE
  LOOP_UNROLL_N(4)    
  for (int i = 0 ; i < N; i++ )
    sum += A[i]*B[i] + C[i];
  
  return sum;
}


int main( int argc, char **argv)
{
  int N = (argc>1? atoi(*(argv+1)) : 1000000 );
  double timing;
  
  printf ( "usage of restrict is %s\n", (RESTRICT?"enabled":"disabled"));
  
  double * A = (double*)malloc( N*sizeof(double) );
  double * B = (double*)malloc( N*sizeof(double) );
  double * C = (double*)malloc( N*sizeof(double) );
  
  timing = PCPU_TIME;
  double sum = process( A, B, C, N );
  timing = PCPU_TIME - timing;  
  printf("\ttiming: %g s, sum is %g\n", timing, sum);


  free ( C ); free ( B ); free ( A );
  
  return 0;
}
