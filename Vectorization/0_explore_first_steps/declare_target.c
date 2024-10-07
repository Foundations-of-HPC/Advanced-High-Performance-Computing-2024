
#pragma GCC target("avx2")
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <cpuid.h>


#ifdef __AVX512__

#warning "found AVX512"
#define V_DSIZE (sizeof( __m512d ) / sizeof(double) )


#elif defined ( __AVX__ ) || defined ( __AVX2__ )

#warning "found AVX/AVX2"
#define V_DSIZE (sizeof( __m256d ) / sizeof(double) )


#elif defined ( __SSE4__ ) || defined ( __SSE3__ )

#warning "found SSE >= 3"
#define V_DSIZE (sizeof( __m128d ) / sizeof(double) )

#else

#warning "no vector capability found"
#define V_DSIZE 1UL

#endif



int main (void )
{
  __builtin_cpu_init();

  printf ( "inquiring the cpu's vendor.. ");
  int eax, ebx, ecx, edx;
  char vendor[13];
  __cpuid(0, eax, ebx, ecx, edx);
  memcpy(vendor, &ebx, 4);
  memcpy(vendor + 4, &edx, 4);
  memcpy(vendor + 8, &ecx, 4);
  vendor[12] = '\0';
  printf ( "%s\n\n", vendor );
  

  printf ( "inquiring cpu's capabilities.. \n");
  if ( __builtin_cpu_supports("avx512f") )
    printf("CPU supports AVX512f\n");
  if ( __builtin_cpu_supports("avx2") )
    printf("CPU supports AVX2\n");
  if ( __builtin_cpu_supports("avx") )
    printf("CPU supports AVX\n");
  if ( __builtin_cpu_supports("sse4.2") )
    printf("CPU supports SSE4.2\n");
  if ( __builtin_cpu_supports("sse4.1") )
    printf("CPU supports SSE4.1\n");
  if ( __builtin_cpu_supports("sse3") )
    printf("CPU supports SSE3\n");

  printf ( "\n" );

  printf ( "we have determined that the double vector size is : %lu\n", V_DSIZE );
    
  return 0;
}
