#include <stdlib.h>
#include <stdio.h>
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */
#define getClock() ((double)clock() / CLOCKS_PER_SEC)

#include <omp.h>

int main ( int argc, char** argv )
{
	int i;       
       	size_t n = 500000000; 

	double A = 16.0; 

	printf ( "\n" );
  	printf ( "The triad stream operation\n" );
  	printf ( "  C/OpenMP version\n" );
  	printf ( "\n" );
	printf( "The total memory allocated is %7.3lf GB.\n",
          3.0*sizeof(double)*n/1024/1024/1024 );
  	      	
	double* D = (double *) malloc( sizeof(double)*n );
      	double* X = (double *) malloc( sizeof(double)*n );
      	double* Y = (double *) malloc( sizeof(double)*n );
  
 	/* ..........Allocate the vector data ............. */
        #pragma omp target data map(X[0:n], Y[0:n]) map(from: D[0:n])
	{
		#pragma omp target teams distribute parallel for
		for ( i=0; i<n; i++)
		{
			X[i] = 1.0;
			Y[i] = 2.0;
		}

	/* ..........function call ............. */
	double tstart = omp_get_wtime();
	#pragma omp target teams num_teams(4) thread_limit(128) distribute parallel for simd
	for ( int i = 0; i < n; i++)
                D[i] = A*X[i] + Y[i];

	double tend = omp_get_wtime();
	printf("Work took (s)= %.6f\n", tend-tstart );
	}

	/* ..........Print the result ............. */
	printf( "\n" );
        for ( i = 0; i < n && i < 10; i++ )
		printf( "  %2d  %10.4f  %10.4f  %10.4f\n", i, X[i], Y[i], D[i] );
	printf ( "\n" );
//	printf("Work took (s)= %.6f\n", tend-tstart );

       
	/* ......Free memory ................. */
        free ( X ); free ( Y ); free ( D );

        /* .......Terminate ........................*/

        return 0;
}
