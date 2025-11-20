#include <stdlib.h>
#include <stdio.h>
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */
#include <openacc.h>
#include <omp.h>

int main ( int argc, char *argv[] )
{
  double start_time, end_time;	
  long long int ARRAY_SIZE = 100000000;	
  
  printf( "The total memory allocated is %7.3lf GB.\n",
          3.0*sizeof(double)*ARRAY_SIZE/1024/1024/1024 );
  
  //double i;
  double A   = 16.0; 

  printf ( "\n" );
  printf ( "Vector addition\n" );
  printf ( "  C/OpenAcc version\n" );
  printf ( "\n" );
  printf ( "  A program which adds two vector.\n" );

  /* ..........Allocate the vector data ............. */
  double *D = (double *) malloc( sizeof(double)*ARRAY_SIZE );
  double *X = (double *) malloc( sizeof(double)*ARRAY_SIZE );
  double *Y = (double *) malloc( sizeof(double)*ARRAY_SIZE );

  // Initization ... 

 #pragma acc parallel loop
  for (size_t i=0; i<ARRAY_SIZE; i++)
  {
          X[i] = 1.0;
          Y[i] = 2.0;
  }
 
  start_time = omp_get_wtime();
  #pragma acc parallel loop
  for ( size_t i=0; i<ARRAY_SIZE; i++ )
	  D[i] = A*X[i] + Y[i];
  end_time = omp_get_wtime();
  
 

  /* ......Print a few entries ................. */
  printf( "\n" );
  printf( "   i        X[i]        Y[i]      D[i] = A*X[i] + Y[i]\n" );
  printf( "\n" );

  for ( size_t i = 0; i < ARRAY_SIZE && i < 10; i++ )
	  printf( "  %2d  %10.4f  %10.4f  %10.4f\n", i, X[i], Y[i], D[i] );
 

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
