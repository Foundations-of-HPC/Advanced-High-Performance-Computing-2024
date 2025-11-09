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
#define VSIZE 32
#include "../headers/vector_types.h"

double process ( double *restrict A, double * restrict B, const int N )
{
  double sum = 0;
  for ( int i = 0; i < N; i++ )
    {      
      double tmp = A[i]*B[i];
      if ( tmp > 0.5 )
	sum += B[i];
    }

  return sum; 
}


int vprocess ( double * restrict A, double * restrict B, const int N )
{
  
  dvector_t *vA  = (dvector_t *)__builtin_assume_aligned(A, VALIGN);
  dvector_t *vB  = (dvector_t *)__builtin_assume_aligned(B, VALIGN);
  dvector_t vsum = {0, 0, 0, 0};
  
  int VN = (N/DVSIZE)&0xFFFFFFFC;
  IVDEP
  LOOP_VECTORIZE
  LOOP_UNROLL_N(4)
  for ( int i = 0; i < VN; i++ )
    {
      dvector_t a     = vA[i]*vB[i];
      dvector_t b     = vB[i];
      llvector_t keep = (a > 0.5);

      vsum += (dvector_t)( (llvector_t)b & keep );
    }

  dvector_u *vsum_u = (dvector_u*)&vsum;
  vsum_u->v[0] += vsum_u->v[1] + (vsum_u->v[2] + vsum_u->v[3]);

  int j = VN*DVSIZE;
  double sum = 0;

  sum += process ( &A[j], &B[j], N-j+1);

  sum += vsum_u->v[0];
  
  return sum; 
}



#define scalar 0
#define vector 1


int main ( int argc, char **argv )
{
  
  int mode   = (argc>1? atoi(*(argv+1)) : scalar );
  int N      = (argc>2? atoi(*(argv+2)) : 1000000 );
  long seed  = (argc>3? atoi(*(argv+3)) : -1 );
  double sum = 0;
  
  double * restrict A = (double*)aligned_alloc( VALIGN, N*sizeof(double) );
  double * restrict B = (double*)aligned_alloc( VALIGN, N*sizeof(double) );

  printf ( "generating random values..\n" );
  if ( seed <0 )
    srand48(time(NULL));
  else
    srand48(seed);
  
  for ( int i = 0; i < N; i++ )
    A[i] = drand48(), B[i] = drand48();

  printf ( "%s processing..\n", (mode==scalar?"scalar":"vector") );
  double timing = PCPU_TIME;
  
  if ( mode == scalar )
    sum = process( A, B, N );
  else
    sum = vprocess( A, B, N );
  
  timing = PCPU_TIME - timing;
  printf ( "result is: %g, processing time: %g sec\n", sum, timing );
  
  free ( B), free (A);
  return 0;
}
