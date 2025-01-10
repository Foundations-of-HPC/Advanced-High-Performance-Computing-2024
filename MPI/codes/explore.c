#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <sched.h>

/*
#include <numa.h>
#include <numaif.h>

#ifdef _OPENMP
#include <omp.h>
#endif
*/

#include <mpi.h>


int get_cpu_id( void );
int read_proc__self_stat( int, unsigned long long int * );

int main ( int argc, char **argv )
{
  int Rank, Ntasks;
  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &Ntasks );
  MPI_Comm_rank( MPI_COMM_WORLD, &Rank );

  char processorname[MPI_MAX_PROCESSOR_NAME+1] = {0};
  int  namelen;
  int  cpuid;
  

  MPI_Get_processor_name( processorname, &namelen );
  cpuid = get_cpu_id();

  printf("Task %d runs on core %d and has processorname %s\n", Rank, cpuid, processorname );

  MPI_Finalize( );


  return 0;
}




/* ««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««««
  
   »»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»»» */


int get_cpu_id( )
{
#if defined(_GNU_SOURCE)                              // GNU SOURCE ------------

  dprintf( 0, "using GNU sched..\n" );
  return  sched_getcpu( );
#else
  
#ifdef SYS_getcpu                                     //     direct sys call ---

  dprintf( 0, "using syscall..\n" );
  int cpuid;
  if ( syscall( SYS_getcpu, &cpuid, NULL, NULL ) == -1 )
    return -1;
  else
    return cpuid;

#else                                                 // NON-GNU UNIX SOURCE ---


  dprintf(0, "using /proc/<pid>/stat\n");
    
  unsigned long long val;
  if ( read_proc_stat( 38, &val ) == -1 )
    return -1;

  return (int)val;

#endif  
#endif                                                // -----------------------


}


int read_proc__self_stat( int field, unsigned long long int *ret_val )
/*
  Interesting fields:

  pid      : 0
  father   : 1
  utime    : 13
  cutime   : 14
  nthreads : 18
  rss      : 22
  cpuid    : 38
 */
{
  // not used, just mnemonic
  char *table[ 52 ] = { [0]="pid", [1]="father", [13]="utime", [14]="cutime", [18]="nthreads", [22]="rss", [38]="cpuid"};
  
  *ret_val = 0;
  
  FILE *file = fopen( "/proc/self/stat", "r" );
  if (file == NULL )
    return -1;

  char   *line;
  int     ret;
  size_t  len;
  ret = getline( &line, &len, file );
  fclose(file);
  
  if( ret == -1 )
    return -1;

  char *token = strtok( line, " ");
  do { token = strtok( NULL, " "); field--; } while( field );
    
  *ret_val = atoll(token);
    
  free(line);
  
  return 0;
}
  

