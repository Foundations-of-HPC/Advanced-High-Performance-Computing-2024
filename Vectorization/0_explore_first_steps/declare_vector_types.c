
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <cpuid.h>


#ifdef __AVX512__

#warning "found AVX512"
typedef __m512d dvector_t;
typedef __m512  fvector_t;
typedef __m512i ivector_t;

#elif defined ( __AVX__ ) || defined ( __AVX2__ )

#warning "found AVX/AVX2"
typedef __m256d dvector_t;
typedef __m256  fvector_t;
typedef __m256i ivector_t;

#elif defined ( __SSE4__ ) || defined ( __SSE3__ )

#warning "found SSE >= 3"
typedef __m128d dvector_t;
typedef __m128  fvector_t;
typedef __m128i ivector_t; 

#else

#warning "no vector capability found"
typedef double dvector_t;
typedef double fvector_t;
typedef double ivector_t;
#endif

#define DV_ELEMENT_SIZE (sizeof( dvector_t ) / sizeof(double) )
#define DV_BIT_SIZE (sizeof( dvector_t ) * 8 )

#define FV_ELEMENT_SIZE (sizeof( fvector_t ) / sizeof(float) )
#define FV_BIT_SIZE (sizeof( fvector_t ) * 8 )

#define IV_ELEMENT_SIZE (sizeof( ivector_t ) / sizeof(int) )
#define IV_BIT_SIZE (sizeof( ivector_t ) * 8 )


int main (void )
{

  printf ( "double vector size is : %lu elements in %lu bits\n",
	   DV_ELEMENT_SIZE, DV_BIT_SIZE );

  printf ( "float vector size is : %lu elements in %lu bits\n",
	   FV_ELEMENT_SIZE, FV_BIT_SIZE );
  
  printf ( "int32 vector size is : %lu elements in %lu bits\n",
	   IV_ELEMENT_SIZE, IV_BIT_SIZE );

  return 0;
}
