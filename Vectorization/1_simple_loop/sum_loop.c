
#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#if !defined(__linux__)
#error "catastrophe, we're not on a linux box!"
#endif

#if defined(__GNU_SOURCE__) || defined(__GNUC__)
#pragma GCC target("avx2")

#elif defined(__clang__)

#pragma clang attribute push (__attribute__((target("avx2"))))
#pragma clang attribute pop

#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
//#include <time.h>

#define INTEGER 0
#define FPSP 1

#if (DTYPE==INTEGER)

#warning "using INT32"
#define dtype uint32_t
#define myRAND lrand48()

#elif (DTYPE==FPSP)

#warning "using FLOAT"
#define dtype float
#define myRAND drand48()

#endif

typedef uint32_t uint;

dtype sum_loop ( dtype *, uint );


dtype sum_loop ( dtype *array, uint N )
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
  
  //srand48( time(NULL) );
  for ( uint i = 0; i < N; i++ )
    array[i] = (dtype)i;
  //array[i] = (dtype)myRAND;

  dtype sum = sum_loop( array, N );

  printf ( "final result is: %g\n", (double) sum );

  free ( array );
  
  return 0;
}


