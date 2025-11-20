#include <stdlib.h>
#include <stdio.h>
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */
#include <openacc.h>
#include <omp.h>

int main ( int argc, char *argv[] )
{
  double start_time, end_time;	
  int n = 20000000;	
  
  printf( "The total memory allocated is %7.3lf GB.\n",
          3.0*sizeof(int)*n/1024/1024/1024 );
  

  //double i;
  double scalar  = 16.0; 

  printf ( "\n" );
  printf ( "Vector addition\n" );
  printf ( "  C/OpenAcc version\n" );
  printf ( "\n" );
  printf ( "  A program which adds two vector.\n" );

  printf("=========================================\n");

  double* restrict a_d = acc_malloc(n * sizeof(double));
  double* restrict b_d = acc_malloc(n * sizeof(double));
  double* restrict c_d = acc_malloc(n * sizeof(double));

  /* ..........Allocate the vector data ............. */
    // Initization ... 

  #pragma acc parallel loop deviceptr(a_d,b_d)
   for (int i=0; i<n; i++) {
      a_d[i] = 1.0;
      b_d[i] = 2.0;
   }

 
  start_time = omp_get_wtime(); 
 
  #pragma acc parallel loop deviceptr(a_d,b_d,c_d)
      for (int i=0; i<n; i++){
         c_d[i] = a_d[i] + scalar*b_d[i];
      }

  end_time = omp_get_wtime();
 

  /* ......Print a few entries ................. */
  printf( "\n" );
  printf( "   i        c[i]\n" );
  printf( "\n" );

  for ( size_t i = 0; i < n && i < 10; i++ )
	  printf( "  %2d %10.4f\n", i, c_d[i] );
 

  printf("\n");

  double time_spent = end_time - start_time;
  printf("time (s)= %.6f\n", time_spent );

  /* ......Free memory ................. */
  
   acc_free(a_d);
   acc_free(b_d);
   acc_free(c_d);

  /* .......Terminate ........................*/
  printf ( "\n" );
  printf ( "Vector addition\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}

/* ..........Program Listing Completed ............. */
