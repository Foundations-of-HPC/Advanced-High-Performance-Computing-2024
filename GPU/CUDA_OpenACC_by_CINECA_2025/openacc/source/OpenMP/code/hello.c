#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

int main() 
{
  int num_devices = omp_get_num_devices();
  printf("Number of available devices on m100 = %d\n", num_devices);
  printf("=======================================\n");
  #pragma omp target 
  {
      if (!omp_is_initial_device()) {
        int nteams= omp_get_num_teams(); 
        int nthreads= omp_get_num_threads();
	printf("Hello World from accelerator\n");
        printf("Running on device with %d teams in total and %d threads in each team\n",nteams,nthreads);
      }
      else
      {
	     printf("Hello World from host\n");
      }
  }
  
}
