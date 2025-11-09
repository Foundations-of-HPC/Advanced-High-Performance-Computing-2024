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

int process ( double * restrict A, double * restrict B, const int N )
{
  IVDEP
  LOOP_VECTORIZE
  LOOP_UNROLL_N(4)
  for ( int i = 0; i < N; i++ )
    {      
      double min = (A[i]>B[i] ? B[i] : A[i]);
      double max = (A[i]>=B[i] ? A[i] : B[i]);
      A[i] = max;
      B[i] = min;
    }
  
  return 0; 
}


int vprocessA ( double * restrict A, double * restrict B, const int N )
{

  dvector_t *vA = (dvector_t *)__builtin_assume_aligned(A, VALIGN);
  dvector_t *vB = (dvector_t *)__builtin_assume_aligned(B, VALIGN);

  
  int VN = (N/DVSIZE)&0xFFFFFFFC;
  IVDEP
  LOOP_VECTORIZE
  LOOP_UNROLL_N(4)
  for ( int i = 0; i < VN; i++ )
    {
      dvector_t a    = vA[i];
      dvector_t b    = vB[i];
      llvector_t keep = (vA[i]>=vB[i]);

      vA[i] = (dvector_t)(((llvector_t)a & keep) | ((llvector_t)b & ~keep));
      vB[i] = (dvector_t)(((llvector_t)b & keep) | ((llvector_t)a & ~keep));
    }

  int j = VN*DVSIZE;
  process ( &A[j], &B[j], N-j+1);
  
  return 0; 
}

int vprocessB ( double * restrict A, double * restrict B, const int N )
{

  dvector_t *vA = (dvector_t *)__builtin_assume_aligned(A, VALIGN);
  dvector_t *vB = (dvector_t *)__builtin_assume_aligned(B, VALIGN);
  
  int VN = (N/DVSIZE)&(-(int)DVSIZE); //&0xFFFFFFFC;
  IVDEP
  LOOP_VECTORIZE
  LOOP_UNROLL_N(4)
  for ( int i = 0; i < VN; i++ )
    {
      dvector_u vAu, vBu;
      vAu.V = vA[i];
      vBu.V = vB[i];

      for ( int j = 0; j < DVSIZE; j++ ) {
	vAu.v[j] = ( vAu.v[j] >= vBu.v[j]? vAu.v[j] : vBu.v[j]);
	vBu.v[j] = ( vAu.v[j] >= vBu.v[j]? vBu.v[j] : vAu.v[j]); }

      vA[i] = vAu.V;
      vB[i] = vBu.V;
    }
  int j = VN*DVSIZE;
  process ( &A[j], &B[j], N-j+1);
  
  return 0; 
}


#define scalar 0
#define vector 1


int main ( int argc, char **argv )
{
  int mode = (argc>1? atoi(*(argv+1)) : scalar );
  int N    = (argc>2? atoi(*(argv+2)) : 1000000 );
  
  double * restrict A = (double*)aligned_alloc( VALIGN, N*sizeof(double) );
  double * restrict B = (double*)aligned_alloc( VALIGN, N*sizeof(double) );

  printf ( "generating random values..\n" );
  srand48(time(NULL));
  for ( int i = 0; i < N; i++ )
    A[i] = drand48(), B[i] = drand48();

  printf ( "%s swapping..\n", (mode==scalar?"scalar":"vector") );
  double timing = PCPU_TIME;
  
  if ( mode == scalar )
    process( A, B, N );
  else switch(mode) {
    case 1: vprocessA( A, B, N ); break;
    case 2: vprocessB( A, B, N ); break; }
  
  timing = PCPU_TIME - timing;
  printf ( "swapping time: %g sec\n", timing );

  printf ( "validating results..\n" );
  int faults = 0;
  for ( int i = 0; i < N; i++ )
    faults += (A[i] < B[i]);

  if ( faults )
    printf ( "oh no, %d entries were misplaced!\n", faults );
  
  free ( B), free (A);
  return 0;
}
