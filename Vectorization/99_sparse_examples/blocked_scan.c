
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "../headers/timing.h"

// baseline (not vectorizable): a[i] += a[i-1]
void scan_baseline(double* a, int n)
{
  for ( int i = 1; i < n; i++ ) a[i] += a[i-1];
}

// vector-friendly blocked scan
// very similar to what we did in HPC basic
// to parallelize the scan with omp threads
//
void scan_blocked(double* a, int n, int B)
{
  
    // 1) partial scans in blocks
    for ( int b = 0; b < n; b += B )
      {
        int end = b+B;
	end = ( (end>n) ? n : end );

        double carry = 0.0;
        for ( int i = b ; i < end; i++)
	  {
	   double x = a[i];
	   a[i] = x + carry;
	   carry = a[i];
	  }
        if(end<n)
	  a[end] += a[end-1]; // stash block total        
      }
    
    // 2) prefix the block totals
    for ( int  b = B; b < n; b+=B )
      a[b] += a[b-B];
    
    // 3) add scanned totals back into each block
    for ( int b = B; b < n; b += B)
      {
        double add = a[b-1];
        int    end = b+B;	
	if(end>n) end = n;
       #pragma omp simd
        for ( int i = b; i < end; i++ )
	  a[i] += add;
      }
}

int main( int argc, char **argv )
{
    int n = (argc>1?atoi(argv[1]):1<<20);
    int B = 256;
    double* a = (double*)aligned_alloc(64, n*sizeof(double));
    
    for ( int i = 0; i < n; i++)
      a[i] = 1.0;

    // warm-up
    scan_baseline(a, n);

    double elapsed = PCPU_TIME;
    scan_baseline(a, n);
    elapsed = PCPU_TIME - elapsed;
    printf("standard: %g s (check: %f)\n",
	   elapsed, a[n-1] );

    
    // reset and run blocked
    for ( int i = 0; i < n; i++ )
      a[i] = 1.0;
    
    scan_blocked(a, n, B);

    elapsed = PCPU_TIME;
    scan_blocked(a, n, B);
    elapsed = PCPU_TIME - elapsed;
    printf("by blocks: %g s (check: %f)\n",
	   elapsed, a[n-1]);
    
    free(a);

    return 0;
}
