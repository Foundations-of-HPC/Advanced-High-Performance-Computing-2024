#include <stdlib.h>
#include <stdio.h>
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */
#define getClock() ((double)clock() / CLOCKS_PER_SEC)

#ifdef _OPENMP
  #include <omp.h>
#endif

int main ( int argc, char** argv )
{       
	size_t n = 500000000;
	int i;
        double d[n], x[n], y[n];
        const int a = 16.0; 

	printf ( "\n" );
  	printf ( "The triad stream operation\n" );
  	printf ( "  C/OpenMP version\n" );
  	printf ( "\n" );
	printf( "The total memory allocated is %7.3lf GB.\n",
          3.0*sizeof(double)*n/1024/1024/1024 );
  	

  	printf ( "\n" );
	int num_devices = omp_get_num_devices();
	int nteams= omp_get_num_teams();
	int nthreads= omp_get_num_threads();

	printf("  Number of available devices   = %d\n", num_devices);
  	printf ( "  Number of processors available = %d\n", nteams );
  	printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );

	printf("Running on device with %d teams in total and %d threads in each team\n",nteams,nthreads);

 	/* ..........Allocate the vector data ............. */
	#pragma omp parallel for
	for ( i=0; i<n; i++)
	{
		d[i] = 0.0;
               	x[i] = 1.0;
               	y[i] = 2.0;
	}
	
	/* ..........function call ............. */
	double tstart = omp_get_wtime();
        #pragma omp target map(to:x,y) map(from:d)
	for (int i = 0; i < n; i++)
		d[i] = a*x[i] + y[i];
	double tstop = omp_get_wtime();

	/* ..........Print the result ............. */
	printf( "\n" );
        for ( i = 0; i < n && i < 10; i++ )
		printf( "  %2d  %10.4f  %10.4f  %10.4f\n", i, x[i], y[i], d[i] );
	printf ( "\n" );
	printf("Work time (s) = %.6f\n", tstop-tstart );


        /* .......Terminate ........................*/

//        return 0;
}
