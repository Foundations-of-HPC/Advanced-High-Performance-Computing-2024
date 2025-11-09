
#include <stdio.h>
#include <stdint.h>

// GCC-specific multiversioning; compile with gcc to see runtime dispatch
//
// https://gcc.gnu.org/onlinedocs/gccint/Target-Attributes.html
//

__attribute__((target_clones("avx512f","avx2","sse4.2","default")))
int dot_i32(const int* a, const int* b, int n)
{
  long long s=0;
  for(int i=0;i<n;i++)
    s += (long long)a[i]*b[i];
  
  return (int)s;
}

int main( void )
{
  int a[32], b[32];
  for(int i=0;i<32;i++)
    {a[i]=i; b[i]=2*i;}
  
  printf("%d\n", dot_i32(a,b,32));
  
  return 0;
}
