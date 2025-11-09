
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#pragma omp declare simd notinbranch simdlen(8) uniform(b)
double f(double a, double b){
    return sin(a) + a*b;
}

double loop_plain(double *x, double b, int n)
{
  double s=0.0;
  for(int i=0;i<n;i++)
    s += f(x[i], b);
  return s;
}

double loop_simd(double *x, double b, int n)
{
  double s=0.0;
 #pragma omp simd reduction(+:s)
  for(int i=0;i<n;i++) s += f(x[i], b);
  return s;
}

int main(int argc, char**argv)
{
  int n = (argc>1?atoi(argv[1]):1000000);
  double b=0.5, *x = (double*)aligned_alloc(64, n*sizeof(double));
  
  for(int i=0;i<n;i++)
    x[i] = i*0.001;
  
  printf("plain: %f\n", loop_plain(x,b,n));
  printf("simd : %f\n", loop_simd(x,b,n));
  
  free(x);

  return 0;
}
