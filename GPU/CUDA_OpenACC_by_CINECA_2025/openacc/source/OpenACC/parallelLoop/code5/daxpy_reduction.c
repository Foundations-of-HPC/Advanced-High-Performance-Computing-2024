#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */
#include <openacc.h>

#define getClock() ((double)clock() / CLOCKS_PER_SEC)

#define ARRAY_SIZE 512*512

int main ( int argc, char *argv[] )
{
  printf( "The total memory allocated is %7.3lf GB.\n",
          3.0*sizeof(double)*ARRAY_SIZE/1024/1024/1024 );
  
  //double i;
  double A   = 16.0; 
  float sum  = 0.0;

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
	  D[i] = 0.0;
          X[i] = 1.0;
          Y[i] = 2.0;
  }
  
  // clock_t begin = clock(); 
  
  double t_start = getClock();
  #pragma acc parallel loop reduction(+:sum)
  for ( size_t i=0; i<ARRAY_SIZE; i++ )
  {
	  D[i] = A*X[i] + Y[i];
  	  sum += D[i];
  }

  double t_end = getClock();

  /* ......Print a few entries ................. */
  printf( "\n" );
  printf( "   i        X[i]        Y[i]      D[i] = A*X[i] + Y[i]\n" );
  printf( "\n" );

  for ( size_t i = 0; i < ARRAY_SIZE && i < 10; i++ )
	  printf( "  %2d  %10.4f  %10.4f  %10.4f\n", i, X[i], Y[i], D[i] );

  double time_spent = t_end - t_start;
  printf("time (s)= %.6f\n", time_spent );
  printf("Reduction sum: %18.16f\n", sum);

  /* ......Free memory ................. */
  free ( X ); free ( Y ); free ( D );

  /* .......Terminate ........................*/
  printf ( "\n" );
  printf ( "Vector addition\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}

/* ..........Program Listing Completed ............. */
