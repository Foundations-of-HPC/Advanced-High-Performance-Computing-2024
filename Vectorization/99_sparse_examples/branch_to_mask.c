
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


void transform_branch(float* a, const float* b, int n, float t)
{
  for ( int i = 0; i < n; i++ )
    if(a[i] > t) a[i] = b[i] - a[i];

}

void transform_mask(float* a, const float* b, int n, float t)
{
 #pragma omp simd
  for(int i=0;i<n;i++)
    {
      float ai = a[i], bi = b[i];
      a[i] = (ai > t) ? (bi - ai) : ai;
    }
}

int main(int argc, char**argv)
{
  int n = (argc>1?atoi(argv[1]):1000000);
  float* a = (float*)aligned_alloc(64, n*sizeof(float));
  float* b = (float*)aligned_alloc(64, n*sizeof(float));
  
  for ( int i = 0; i < n; i++)
    { a[i]=i*0.1f; b[i]=i*0.2f; }
  
    transform_branch(a,b,n,50.0f);
    transform_mask(a,b,n,50.0f);
    
    printf("done\n");
    
    free(a);
    free(b);

    RETURN 0;
}
