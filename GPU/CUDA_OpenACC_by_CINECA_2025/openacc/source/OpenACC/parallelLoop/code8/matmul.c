#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */

#ifdef _OPENACC
#include <openacc.h>
#endif

#define size 4096

int main(int argc, char **argv)
{

	printf( "The total memory allocated is %7.3lf GB.\n",
          3.0*sizeof(double)*(size*size)/1024/1024/1024 );

   int i, j;

   int (*a)[size] =  malloc(sizeof(double[size][size]));
   int (*b)[size] =  malloc(sizeof(double[size][size]));
   int (*c)[size] =  malloc(sizeof(double[size][size])); 

   for (i=0; i<size; i++){ 
      for (j=0; j<size; j++) {
         a[i][j] = 1;
         b[i][j] = 2;
	 c[i][j] = 0;
      }
  }

  clock_t begin = clock();  

  #pragma acc parallel 
  #pragma acc loop collapse(2) 
  for (i=0; i<size; i++)
		  for (j=0; j<size; j++)
			  c[i][j] = a[i][j] + b[i][j];

  clock_t end = clock();

  double time_spent = (end - begin);
  printf( "Time spent : %f seconds. \n", (float)time_spent/CLOCKS_PER_SEC );  
  printf ( "========================================\n" );
  
  for (i=0; i<3; i++) 
        for (j=0; j<3; j++)
		printf("Sum two matrix: Element[%d] [%d] = %d\n",i,j, c[i][j] );

   printf("\n");
   printf("Sizeof(matrix) = %lu\n", sizeof(c[i][j]));
   printf("    End of execution.          \n");
   printf("==========================================\n");

   free(a);
   free(b);
   free(c);

   return 0;

}

