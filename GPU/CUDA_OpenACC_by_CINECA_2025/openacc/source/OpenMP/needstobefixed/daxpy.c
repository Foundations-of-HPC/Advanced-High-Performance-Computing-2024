#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#ifdef _OPENMP
 #include<omp.h>
#else
 #define omp_get_num_threads() 2
#endif

#define n 256*1024*256

int main ( int argc, char *argv[] )
{
  /* ..........Allocate the vector data ............. */

  #pragma omp declare target
   double *a, *b, *c;
  #pragma omp end declare target


   #pragma omp target
   {
       a = malloc(n* sizeof(double));
       b = malloc(n* sizeof(double));
       c = malloc(n* sizeof(double));
   }

     double scalar = 16.0;

    // initialize on the host
    #pragma omp target teams distribute parallel for simd is_device_ptr(a, b)
    for (size_t i=0; i<n; i++)
    {
      a[i] = 2.0;
      b[i] = 1.0;
    }

   double tick = omp_get_wtime();
  #pragma omp target teams distribute parallel for simd is_device_ptr(a, b, c)
  for ( size_t i=0; i<n; i++ )
  {
    c[i] = a[i] + scalar*b[i];
  }

  double tock = omp_get_wtime();
 
/*
  Print a few entries.
*/
  printf ( "\n" );
  printf ( "   i        a[i]        b[i]      c[i] = a[i] + b[i]\n" );
  printf ( "\n" );
  for ( size_t i = 0; i < n && i < 10; i++ )
  {
    printf ( "  %2d  %10.4f  %10.4f  %10.4f\n", i, a[i], b[i], c[i] );
  }
/*
  Free memory.
*/


  #pragma omp target
   {
       free(a);
       free(b);
       free(c);
   }

/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "Vector addition\n" );
  printf ( "  Normal end of execution.\n" );

   printf("===================================== \n");
    printf("Work took %f seconds\n", tock - tick);
   printf("===================================== \n");

  return 0;
}

/* ..........Program Listing Completed ............. */

