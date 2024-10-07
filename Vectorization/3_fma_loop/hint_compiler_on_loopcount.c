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


double __attribute__ ((noinline)) process ( double *restrict A, double *restrict B, double *restrict C, int N )
{
  double sum     = 0;
  double asum[4] = {0};
 #if !defined(HINT)
  int N4         = ((N/4)*4);
 #else
  int N4         = N&&0xFFFFFFFC;
    // alternative form N_4 __attribute((__assume(N_4 % 4 == 0)));
 #endif
   
  IVDEP
  LOOP_VECTORIZE
    for (int i = 0 ; i < N4; i+=4 )
    {
      asum[0] += A[i  ]*B[i  ] + C[i  ];
      asum[1] += A[i+1]*B[i+1] + C[i+1];
      asum[2] += A[i+2]*B[i+2] + C[i+2];
      asum[3] += A[i+3]*B[i+3] + C[i+3];
    }
  asum[0] += asum[1] + (asum[2] + asum[3]);
  
  for (int i = N4 ; i < N; i++ )
    sum += A[i]*B[i] + C[i];

  sum += asum[0];

  return sum;
}



int main( int argc, char **argv)
{  
  int N = (argc>1? atoi(*(argv+1)) : 1000000 );
  double timing;

  double * restrict A = (double*)malloc( N*sizeof(double) );
  double * restrict B = (double*)malloc( N*sizeof(double) );
  double * restrict C = (double*)malloc( N*sizeof(double) );

  timing = PCPU_TIME;
  double sum = process( A, B, C, N );
  timing = PCPU_TIME - timing;  
  printf("\ttiming: %g s, sum is %g\n", timing, sum);

  free ( C ); free ( B ); free ( A );
  
  return 0;
}
