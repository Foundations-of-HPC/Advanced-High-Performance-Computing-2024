#include <stdlib.h>
#include <stdio.h>
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */
#define getClock() ((double)clock() / CLOCKS_PER_SEC)

#include <omp.h>

void daxpygpu( size_t n,
	       float a, 
	       double *x,
	       double *restrict y ) {
	#pragma acc kernels
        for ( int i = 0; i < n; i++)	
	        y[i] = a*x[i] + y[i];
}

int main ( int argc, char** argv )
{
	int i; 
        size_t n = 1<<29;
	float a = 16.0;
    
	printf ( "\n" );
  	printf ( "The triad stream operation\n" );
  	printf ( "  C/OpenMP version\n" );
  	printf ( "\n" );
	printf( "The total memory allocated is %7.3lf GB.\n",
          2.0*sizeof(double)*n/1024/1024/1024 );
  	      	
      	double* x = (double *) malloc( sizeof(double)*n );
      	double* y = (double *) malloc( sizeof(double)*n );
  
 	/* ..........Allocate the vector data ............. */
       
	for ( i=0; i<n; i++)
	{
               	x[i] = 1.0;
               	y[i] = 2.0;
	}
	
	/* ..........function call ............. */
	double tstart = omp_get_wtime();
	daxpygpu(n, a, x, y);
	double tend = omp_get_wtime();

	/* ..........Print the result ............. */
	printf( "\n" );
        for ( i = 0; i < n && i < 10; i++ )
		printf( "  %2d  %10.4f  %10.4f \n", i, x[i], y[i] );
	printf ( "\n" );
	printf("Work took (s)= %.6f\n", tend-tstart );

       
	/* ......Free memory ................. */
        free ( x ); free ( y );

        /* .......Terminate ........................*/

        return 0;
}
