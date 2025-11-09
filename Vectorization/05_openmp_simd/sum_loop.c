
#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#if !defined(__linux__)
#error "catastrophe, we're not on a linux box!"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "../headers/timing.h"


double sum_loop ( double * restrict, const int );

double sum_loop ( double * restrict array, const int N )
{
  double sum = 0;
 #pragma omp simd reduction(+:sum)
  for ( int i = 0; i < N; i++ )
    sum += array[i];
  
  return sum;
}                                       


int main ( int argc, char **argv )
{

  int N = ( (argc > 1)? (int)atoi(*(argv+1)) : 1000000 );

  double *array = (double*)malloc( sizeof(double) * N );
  
  for ( int i = 0; i < N; i++ )
    array[i] = (double)i;

  double timing = PCPU_TIME;
  double sum = sum_loop( array, N );
  timing = PCPU_TIME - timing;

  printf ( "timing: %g, final result is: %g\n", timing, (double) sum );

  free ( array );
  
  return 0;
}


