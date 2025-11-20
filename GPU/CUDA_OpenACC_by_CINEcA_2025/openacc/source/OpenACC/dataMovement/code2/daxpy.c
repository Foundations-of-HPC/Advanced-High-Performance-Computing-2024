#include <stdlib.h>
#include <stdio.h>
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */
#include <openacc.h>
#include <omp.h>

int main ( int argc, char *argv[] )
{
  double start_time, end_time;	
  long long int n = 600000000;	
  
  printf( "The total memory allocated is %7.3lf GB.\n",
          3.0*sizeof(double)*n/1024/1024/1024 );
  

  //double i;
  double A   = 16.0; 

  printf ( "\n" );
  printf ( "Vector addition\n" );
  printf ( "  C/OpenAcc version\n" );
  printf ( "\n" );
  printf ( "  A program which adds two vector.\n" );

  printf("=========================================\n");

  /* ..........Allocate the vector data ............. */
  double *D = (double *) malloc( sizeof(double)*n );
  double *X = (double *) malloc( sizeof(double)*n );
  double *Y = (double *) malloc( sizeof(double)*n );

  // Initization ... 

  #pragma acc parallel loop  
  for (size_t i=0; i<n; i++)
  {
	  X[i] = 1.0;
          Y[i] = 2.0;
  }

  start_time = omp_get_wtime(); 
   #pragma acc enter data copyin( X[0:n], Y[0:n] ) create(D[0:n])
  {
  #pragma acc parallel loop
  for ( size_t i=0; i<n; i++ )
	  D[i] = A*X[i] + Y[i];
  }
  end_time = omp_get_wtime();

  #pragma acc exit data copyout(D[0:n]) delete(X,Y)
 

  /* ......Print a few entries ................. */
  printf( "\n" );
  printf( "   i        D[i] = A*X[i] + Y[i]\n" );
  printf( "\n" );

  for ( size_t i = 0; i < n && i < 10; i++ )
	  printf( "  %2d %10.4f\n", i, D[i] );
 

  printf("\n");

  double time_spent = end_time - start_time;
  printf("time (s)= %.6f\n", time_spent );

  /* ......Free memory ................. */
  free ( X ); free ( Y ); free ( D );

  /* .......Terminate ........................*/
  printf ( "\n" );
  printf ( "Vector addition\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}

/* ..........Program Listing Completed ............. */
