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


/*
 *  ALL conditional compilations options
 *
 * ALIGN_MEMORY     memory is allocated aligned, __assume__ hints are given
 * UNALIGN_MEMORY   memory explicitly misaligned
 * WARMUP_MEMORY    warm-up before calling the kernel
 *
 * CASE 0           "normal" scalar loop
 * CASE 1           scalar loop unrolled
 * CASE 2           vector loop
 */


#define ALIGN 32

#if !defined(CASE)
#define CASE 0
#endif

#if (CASE == 2) && !defined(ALIGN_MEMORY)
#warning "defining ALIGN_MEMORY by default when using vector types"
#define ALIGN_MEMORY
#endif

#if (CASE == 2) && defined(UNALIGN_MEMORY)
#error "it is not allowed to run CASE=2 with UNALIGN_MEMORY"
#endif

#if defined(ALIGN_MEMORY) && defined(UNALIGN_MEMORY)
#error "you can have either aligned or unaligned memory, not both"
#endif


void __attribute__ ((noinline)) warmup( double *restrict A, double *restrict B, double *restrict C, int N )
{
 #if defined(ALIGN_MEMORY) 
  A = __builtin_assume_aligned(A, ALIGN);
  B = __builtin_assume_aligned(B, ALIGN);
  C = __builtin_assume_aligned(C, ALIGN);
 #endif

  IVDEP
  LOOP_VECTORIZE
  LOOP_UNROLL_N(4)
  for ( int i = 0; i < N; i++ )
    A[i] = 1.0, B[i] = 1.0, C[i] = 1.0;
  
  /* for ( int i = 0; i < N; i++ ) */
  /*   B[i] = 1.0; */
  /* for ( int i = 0; i < N; i++ ) */
  /*   C[i] = 1.0; */

  return;
}




double __attribute__ ((noinline)) process ( const double *restrict A, const double *restrict B, const double *restrict C, int N )
{
 #if defined(ALIGN_MEMORY)
  A = __builtin_assume_aligned(A, ALIGN);
  B = __builtin_assume_aligned(B, ALIGN);
  C = __builtin_assume_aligned(C, ALIGN);
 #endif

  double sum =0;
  
  // ····································
 #if CASE == 0
  IVDEP
  LOOP_VECTORIZE
  LOOP_UNROLL_N(4)    
  for (int i = 0 ; i < N; i++ )
    sum += A[i]*B[i] + C[i];

  // ····································
 #elif CASE == 1

  double asum[4] = {0};
  int i          = 0;
  int N_4        = ((N/4)*4)&0xFFFFFFFC;
  /*
  N_4 = N&0xFFFFFFFC;
  //N_4 __attribute((__assume(N_4 % 4 == 0)));
  */
  IVDEP
  LOOP_VECTORIZE
  for ( ; i < N_4; i+=4 )
    {
      asum[0] += A[i  ]*B[i  ] + C[i  ];
      asum[1] += A[i+1]*B[i+1] + C[i+1];
      asum[2] += A[i+2]*B[i+2] + C[i+2];
      asum[3] += A[i+3]*B[i+3] + C[i+3];
    }
  asum[0] += asum[1] + (asum[2] + asum[3]);
  
  for ( ; i < N; i++ )
    sum += A[i]*B[i] + C[i];

  sum += asum[0];


 #elif CASE == 2

 #define VD_SIZE 4
  
  typedef double v4df __attribute__ ((vector_size (VD_SIZE*sizeof(double))));
  typedef union {
    v4df   V;
    double v[VD_SIZE];
  }v4df_u;

 #if defined(ALIGN_MEMORY)
  v4df *vA = (v4df *)__builtin_assume_aligned(A, ALIGN);
  v4df *vB = (v4df *)__builtin_assume_aligned(B, ALIGN);
  v4df *vC = (v4df *)__builtin_assume_aligned(C, ALIGN);
 #else
  v4df *vA = (v4df *)A;
  v4df *vB = (v4df *)B;
  v4df *vC = (v4df *)C;
 #endif
  v4df vsum = {0};
  int N4  = (N/4)&0xFFFFFFFC;;

  for ( int i = 0 ; i < N4; i++ )
    vsum += vA[i] * vB[i] + vC[i];

  v4df_u *vsum_u = (v4df_u*)&vsum;
  vsum_u->v[0] += vsum_u->v[1] + (vsum_u->v[2] + vsum_u->v[3]);
  
  for ( int i = N4*4; i < N; i++ )
    sum += A[i]*B[i] + C[i];

  sum += vsum_u->v[0];
 #endif
  
  return sum;
}



int main( int argc, char **argv)
{
  char buffer[500];
  sprintf( buffer, "case %d :: ", CASE );
 #if defined(WARMUP_MEMORY)
  sprintf( &buffer[strlen(buffer)], "warmup memory :: " );
 #endif
 #if defined(ALIGN_MEMORY)
  sprintf( &buffer[strlen(buffer)], "aligned memory :: " );
 #endif
 #if defined(UNALIGN_MEMORY)
  sprintf( &buffer[strlen(buffer)], "unaligned memory :: " );
 #endif

  printf ( "%s\n", buffer );
  
  int N = (argc>1? atoi(*(argv+1)) : 1000000 );
  double timing;

  
 #if defined(ALIGN_MEMORY)
  double * restrict A = (double*)aligned_alloc( ALIGN, N*sizeof(double) );
  double * restrict B = (double*)aligned_alloc( ALIGN, N*sizeof(double) );
  double * restrict C = (double*)aligned_alloc( ALIGN, N*sizeof(double) );

 #else
 #if defined(UNALIGN_MEMORY)
 #define _N (N+1)
 #else
 #define _N N
 #endif  
  double * restrict A = (double*)malloc( _N*sizeof(double) );
  double * restrict B = (double*)malloc( _N*sizeof(double) );
  double * restrict C = (double*)malloc( _N*sizeof(double) );
 #if defined(UNALIGN_MEMORY)
  double *_A = A;
  double *_B = B;
  double *_C = C;
  
  A = (double*)((char*)A+1);
  B = (double*)((char*)B+1);
  C = (double*)((char*)C+1);
 #endif
 #endif
  
 #if defined(WARMUP_MEMORY)
  timing = PCPU_TIME;
  warmup( A, B, C, N );
  timing = PCPU_TIME - timing;
  printf("\tmemory initialization time: %g\n", timing);
 #endif

  timing = PCPU_TIME;
  double sum = process( A, B, C, N );
  timing = PCPU_TIME - timing;  
  printf("\ttiming: %g s, sum is %g\n", timing, sum);


 #if defined(UNALIGN_MEMORY)
  free ( _C ); free( _B ); free( _A );
 #else
  free ( C ); free ( B ); free ( A );
  #endif
  
  
  return 0;
}
