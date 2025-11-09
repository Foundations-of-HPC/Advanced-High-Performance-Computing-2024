
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>

#include "../headers/timing.h"

void daxpy_aliasing(int n, double a, double *x, double *y)
{
  for ( int i = 0; i < n; i++)
    y[i] = a*x[i] + y[i];
}

void daxpy_norestrict(int n, double a, double *x, double *y)
{
 #pragma omp simd
  for ( int i = 0; i < n; i++ )
    y[i] = a*x[i] + y[i];
}

void daxpy_restrict(int n, double a, double * __restrict x, double * __restrict y)
{
 #pragma omp simd
  for ( int i = 0; i < n; i++ )
    y[i] = a*x[i] + y[i];
}

int main(int argc, char **argv)
{
  int n = (argc>1?atoi(argv[1]):1000000);
  
  double *x,*y;
  posix_memalign((void**)&x,64,n*sizeof(double));
  posix_memalign((void**)&y,64,n*sizeof(double));
  
  for(int i=0;i<n;i++) {
    x[i] = 1.0; y[i] = 2.0; }
  

  double elapsed = PCPU_TIME;
  daxpy_aliasing(n, 2.0, x, y);
  elapsed = PCPU_TIME - elapsed;
  printf ("aliasing: %g s\n", elapsed);

  elapsed = PCPU_TIME;
  daxpy_norestrict(n, 2.0, x, y);
  elapsed = PCPU_TIME - elapsed;
  printf ("simd, no restrict: %g s\n", elapsed);

  elapsed = PCPU_TIME;
  daxpy_restrict(n, 2.0, x, y);
  elapsed = PCPU_TIME - elapsed;
  printf ("simd, restrict: %g s\n", elapsed);
  
  printf("OK\n");
  
  free(x);
  free(y);
  
  return 0;
}
