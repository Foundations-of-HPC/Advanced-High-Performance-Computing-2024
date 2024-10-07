
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

uint loop ( uint *restrict a, const uint *restrict c, const uint N )
{
  uint sum = 0;
  
  a[0] = c[0];
  for ( uint i = 1; i < N; i++ ) {
    a[i] = a[i-1] + c[i];
    sum += a[i]; }
  
  return sum;
}                                       



int main ( int argc, char **argv )
{

  uint N      = ( (argc > 1)? (uint)atoi(*(argv+1)) : 1000000 );
  
  uint *a = (uint*)malloc( sizeof(uint) * N );
  uint *c = (uint*)malloc( sizeof(uint) * N );
  
  //srand48( time(NULL) );
  for ( uint i = 0; i < N; i++ ) {
    a[i] = (uint)i;
    c[i] = (uint)(i*2); }
  //array[i] = (uint)myRAND;

  uint sum = loop( a, c, N );
  printf ( "final result is: %u\n", sum );

  free ( c );
  free ( a );
  
  return 0;
}


