
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
#include <time.h>

#if !defined(DTYPE) || (DTYPE==int) || (DTYPE==INT)
#define DTYPE uint32_t
#define myRAND lrand48()
#else
#define DTYPE double
#define myRAND drand48()
#endif

typedef uint32_t uint;

DTYPE sum_loop ( DTYPE *, uint );


DTYPE sum_loop ( DTYPE *array, uint N )
{
  DTYPE sum = 0;
  for ( uint32_t i = 0; i < N; i++ )
    sum += array[i];

  return sum;
}


int main ( int argc, char **argv )
{

  uint32_t N = ( (argc > 1)? (uint)atoi(*(argv+1)) : 1000000 );

  DTYPE *array = (DTYPE*)malloc( sizeof(DTYPE) * N );
  
  srand48( time(NULL) );
  for ( uint i = 0; i < N; i++ )
    array[i] = (DTYPE)myRAND;

  DTYPE sum = sum_loop( array, N );

  printf ( "final result is: %g\n", (double) sum );
  
  return 0;
}


