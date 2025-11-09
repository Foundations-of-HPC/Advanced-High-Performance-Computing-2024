
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
#include <stdint.h>
#include "../headers/timing.h"

#define INTEGER 0
#define FPDP 1

#if (DTYPE==INTEGER)

#warning "using INT32"
#define myRAND lrand48()
#define dtype uint32_t

#elif (DTYPE==FPDP)

#warning "using FLOAT"
#define myRAND drand48()
#define dtype double

#else

#error "unknown DTYPE"
#endif

typedef uint32_t uint;

dtype sum_loop ( dtype * restrict, const uint );

dtype sum_loop ( dtype * restrict array, const uint N )
{
  dtype sum = 0;
  for ( uint32_t i = 0; i < N; i++ )
    sum += array[i];
  return sum;
}                                       


int main ( int argc, char **argv )
{

  uint32_t N = ( (argc > 1)? (uint)atoi(*(argv+1)) : 1000000 );

  dtype *array = (dtype*)malloc( sizeof(dtype) * N );
  
  for ( uint i = 0; i < N; i++ )
    array[i] = (dtype)i;

  double timing = PCPU_TIME;
  dtype sum = sum_loop( array, N );
  timing = PCPU_TIME - timing;

  printf ( "timing: %g, final result is: %g\n", timing, (double) sum );

  free ( array );
  
  return 0;
}


