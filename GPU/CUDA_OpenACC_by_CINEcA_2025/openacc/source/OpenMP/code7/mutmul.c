/* ======================================================
 * 2D Vector addition
 * The value of 3 for each element 
 * Nitin Shukla
 * =====================================================*/

#include <stdio.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define N 2048
#define M 2048
double A[N][M], B[N][M], C[N][M];

void initMatrix(double *A, int n, int m, double c) {
  int i,j;

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      A[i*m+j] = c;
}

void computeonGPU()
{
	int i, j; 
	#pragma omp target teams distribute parallel for simd map(to: A, B) map(tofrom: C) num_teams(128) thread_limit(1024) //private(i,j)
        for (i=0; i<N; i++)
                for (j=0; j<M; j++)
                         C[i][j] = A[i][j] + B[i][j];
}

int main(int argc, char *argv[]) {
  int i, j; 
  double time2;


  printf ( "\n" );
  printf ( "2D vector sum\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "==========================================================\n" );

  printf ( "  Number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  Number of threads =              %d\n", omp_get_max_threads ( ) );

  printf( "The total memory allocated is %7.3lf GB.\n",
          3.0*sizeof(double)*(N*M)/1024/1024/1024 );

  printf ( "==========================================================\n" );

  initMatrix((double *) A, N, M, 1.0);
  initMatrix((double *) B, N, M, 2.0);
  initMatrix((double *) C, N, M, 0.0);

  // C = A + B

#ifdef _OPENMP
    double time1 = omp_get_wtime();
#else
    time_t time1 = clock();
#endif

    computeonGPU();

#ifdef _OPENMP
    time2 = omp_get_wtime() - time1;
#else
    time2 = (clock() - time1) / (double) CLOCKS_PER_SEC;
#endif

    printf ( "\n");
    printf("  Elapsed time (s) = %.9lf\n", time2);
    printf ( "==========================================================\n" );

  for (i=0; i<3; i++)
	  for (j=0; j<3; j++)
		    printf("%f %f %f\n", C[0][0], C[1][1], C[N-1][M-1]);

  return 0;
}


