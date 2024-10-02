
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
//#include <time.h>

typedef unsigned int uint;

uint sum_loop ( uint *array, uint N, int stride )
{
  uint sum = 0;
  for ( uint i = 0; i < N; i+=stride )
    sum += array[i];
  return sum;
}                                       


int main ( int argc, char **argv )
{

  uint N      = ( (argc > 1)? (uint)atoi(*(argv+1)) : 1000000 );
  int  stride = ( (argc > 2)? atoi(*(argv+2)) : 1 );
  
  uint *array = (uint*)malloc( sizeof(uint) * N );
  
  //srand48( time(NULL) );
  for ( uint i = 0; i < N; i++ )
    array[i] = (uint)i;
  //array[i] = (uint)myRAND;

  uint sum = sum_loop( array, N, stride );

  printf ( "final result is: %u\n", sum );

  free ( array );
  
  return 0;
}


