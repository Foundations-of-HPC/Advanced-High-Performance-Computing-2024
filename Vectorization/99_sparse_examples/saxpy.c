
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#include "../headers/timing.h"


static void saxpy_baseline(int n, float a, float *x, float *y, float *out)
{
  for ( int i = 0; i < n; i++ )
    out[i] = a*x[i] + y[i];
}

static void saxpy_restrict(int n, float a,
                           float * __restrict x,
                           float * __restrict y,
                           float * __restrict out)
{
 #pragma omp simd
  for (int i=0;i<n;i++) out[i] = a*x[i] + y[i];
}

static void saxpy_assume_aligned(int n, float a,
                                 float * __restrict x,
                                 float * __restrict y,
                                 float * __restrict out)
{
  float *X = (float*)__builtin_assume_aligned(x, 64);
  float *Y = (float*)__builtin_assume_aligned(y, 64);
  float *O = (float*)__builtin_assume_aligned(out, 64);
 #pragma omp simd
  for (int i=0;i<n;i++) O[i] = a*X[i] + Y[i];
}

int main(int argc, char **argv){
    int n = (argc>1?atoi(argv[1]):(1<<20));
    float *x, *y, *out;
    
    if (posix_memalign((void**)&x, 64, n*sizeof(float)) ||
        posix_memalign((void**)&y, 64, n*sizeof(float)) ||
        posix_memalign((void**)&out,64, n*sizeof(float)))
      {
	printf("memory allocaiton failed\n");
	return 1;
      }
    
    for (int i=0;i<n;i++)
      { x[i]=i*0.5f; y[i]=i*0.25f; }

    double elapsed;
    int idx;
    sdrand48();
    
    saxpy_baseline(n, 2.0f, x, y, out);
    
    elapsed = PCPU_TIME;
    saxpy_baseline(n, 2.0f, x, y, out);
    elapsed = PCPU_TIME - elapsed;
    printf("base: %g s (%f)\n",
	   elapsed, out[lrand48()%n] );

    elapsed = PCPU_TIME;
    saxpy_restrict(n, 2.0f, x, y, out);
    elapsed = PCPU_TIME - elapsed;
    printf("restrict: %g s (%f)\n",
	   elapsed, out[lrand48()%n] );

    elapsed = PCPU_TIME;
    saxpy_assume_aligned(n, 2.0f, x, y, out);
    elapsed = PCPU_TIME - elapsed;
    printf("assume aligned: %g s (%f)\n",
	   elapsed, out[lrand48()%n] );

    free(x);
    free(y);
    free(out);

    return 0;
}
