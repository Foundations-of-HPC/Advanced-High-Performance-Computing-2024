
#ifdef __AVX512F__
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

double dot_avx512_masked(const double* a, const double* b, int n)
{
  __m512d acc = _mm512_setzero_pd();
  int i=0;
  for( ; i+8 <= n; i+= 8)
    {
      __m512d va = _mm512_load_pd(a+i);
      __m512d vb = _mm512_load_pd(b+i);
      acc = _mm512_fmadd_pd(va, vb, acc);
    }
  
    int rem = n - i;
    if(rem){
        __mmask8 k = (1u<<rem)-1u;
        __m512d va = _mm512_maskz_loadu_pd(k, a+i);
        __m512d vb = _mm512_maskz_loadu_pd(k, b+i);
        acc = _mm512_mask3_fmadd_pd(va, vb, acc, k);
    }
    return _mm512_reduce_add_pd(acc);
}

int main(int argc, char**argv)
{
    int n = (argc>1?atoi(argv[1]):1000003);
    double* a = (double*)aligned_alloc(64, n*sizeof(double));
    double* b = (double*)aligned_alloc(64, n*sizeof(double));
    
    for ( int i = 0; i < n; i++ ) {
      a[i]=1.0; b[i]=2.0; }
    
    printf("%f\n", dot_avx512_masked(a,b,n));
    
    free(a);
    free(b);

    return 0;
}

#else

#error "AVX512f not available on this target"

#endif
