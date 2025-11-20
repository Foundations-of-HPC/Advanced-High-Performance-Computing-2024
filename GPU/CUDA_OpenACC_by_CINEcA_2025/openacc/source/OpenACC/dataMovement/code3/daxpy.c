#include <stdlib.h>
#include <stdio.h>
#include <time.h>    /* clock_t, clock, CLOCKS_PER_SEC */
#include <openacc.h>
#include <omp.h>

int* allocate_array(int N)
{
	int* ptr = (int *) malloc(N * sizeof(int));
	#pragma acc enter data create(ptr[0:N]) 
	return ptr;
}

void deallocate_array(int* ptr){
  #pragma acc exit data delete(ptr)
  free(ptr);
}

int main()
{
	int* a = allocate_array(100);
  	#pragma acc kernels
	{
		a[0] = 0;
	}
	
	deallocate_array(*a); 

}


/* ..........Program Listing Completed ............. */
