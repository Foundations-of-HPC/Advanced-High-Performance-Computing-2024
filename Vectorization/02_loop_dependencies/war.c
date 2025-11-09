
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

typedef unsigned int uint;

void loop_init ( uint *restrict a, const uint *restrict c, const uint N )
{
  for ( uint i = 1; i < N; i++ )
    a[i-1] = a[i] + c[i];
  
  return;
}                                       


uint loop_sum ( uint *restrict a, const uint *restrict c, const uint N )
{
  uint sum = a[0];
  for ( uint i = 1; i < N; i++ ) {
    a[i-1] = a[i] + c[i];
    sum += a[i]; }
  
  return sum;
}                                       

uint loop_sum_reshuffle ( uint *restrict a, const uint *restrict c, const uint N )
{

  uint sum = 0;
  // Process the bulk with vectorizable loop
  // We can safely look ahead by 4 only up to N-4
  uint vec_limit = (N > 4) ? N - 4 : 0;

  {
    // Initial sum of first 4 elements (or all if N < 4)
    uint init_limit = (N < 4) ? N : 4;
    for ( uint i = 1; i < init_limit; i++ )
      sum += a[i];
  }

 #pragma GCC ivdep
  for ( uint i = 1; i < vec_limit; i++ ) {
    a[i-1] = a[i] + c[i];
    sum += a[i+4]; }
  
  // Handle tail: elements from vec_limit to N-1
  for ( uint i = vec_limit; i < N; i++ )
    {
      if (i > 0)
	a[i-1] = a[i] + c[i];
    }
  
  return sum;
}                                       



int main ( int argc, char **argv )
{

  uint N      = ( (argc > 1)? (uint)atoi(*(argv+1)) : 100 );
  
  uint *a = (uint*)malloc( sizeof(uint) * (N+4) );
  uint *c = (uint*)malloc( sizeof(uint) * N );

  // ·············································
  // set-up
  //
  for ( uint i = 0; i < N; i++ ) {
    a[i] = (uint)i;
    c[i] = (uint)(i*2); }

  for ( uint i = N; i < N+4; i++ )
    a[i] = 0;

  // ············································
  // process a[] with a vectorizable loop
  //
  loop_init( a, c, N );

  // ············································
  // ripristinate a[]
  //
  for ( uint i = 0; i < N; i++ )
    a[i] = (uint)i;

  // ············································
  // process a[] with a non-vectorizable loop
  //
  uint sum = loop_sum( a, c, N );
  printf ( "result of non-vec loop: %u\n", sum );

  // ············································
  // ripristinate a[]
  //
  for ( uint i = 0; i < N; i++ )
    a[i] = (uint)i;

  // ············································
  // process a[] with a vectorizable loop
  // the result returned is the same than for the
  // non-vectorizable loop
  //
  sum = loop_sum_reshuffle( a, c, N );
  printf ( "result of vec-loop: %u\n", sum );

  free ( c );
  free ( a );
  
  return 0;
}


