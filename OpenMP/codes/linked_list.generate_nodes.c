
#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>


// =========================================================================
//
//  definees
//


// ··········································.
//  types

typedef unsigned           int uint;
typedef unsigned long long int ull;



// ··········································.
//  timing
//

#if defined(_OPENMP)

// get the wall-clock time
#define CPU_TIME ({ struct timespec ts; (clock_gettime( CLOCK_REALTIME, &ts ), \
					 (ull)ts.tv_sec * 1000000000 +	\
					 (ull)ts.tv_nsec); })

// get per-thread cpu time
#define CPU_TIME_th ({ struct  timespec ts; (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &ts ), \
					     (ull)ts.tv_sec*1000000000 + \
					     (ull)ts.tv_nsec); })

#else

// get per-proc cpu time
#define CPU_TIME ({ struct timespec ts; (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), \
					 (ull)ts.tv_sec * 1000000000 +	\
					 (ull)ts.tv_nsec); })
#endif

// ··········································.
// get a timestamp for debugging purposes
//
#if defined(DEBUG)
#define TIME_CUT 100000000
#define TIMESTAP (CPU_TIME % TIME_CUT)
#define dbgout(...) printf( __VA_ARGS__ );
#else
#define TIMESTAP
#define dbgout(...) 
#endif


char filename_dflt[] = "nodes.dat";



int main ( int argc, char **argv )
{
  
  // options controlling how the nodes are generated
  //
  uint     N;            // how many nodes to generate

  int      mode;         //  0 = random, the max vale will be the value of arg "cutoff"
                         //  1 = random flucuations around the value of arg "cutoff"
                         //      the additional argument "dispersion" specifies the
                         //      amplitude of the fluctuation
  uint     cutoff;
  float    dispersion;
  long int seed;         // if needed

  char     *filename;
  ull       timing;
  

  {
    // parse the command line arguments
    // let's use getopt since we use a simple command line
  
    N              = 10000;
    mode           = 0;
    cutoff         = 500;
    dispersion     = 0.1;
    seed           = 12345;
    
    filename       = filename_dflt;
    
    int c;
    while ( (c = getopt(argc, argv, "N:m:c:d:s:f:h"))  != -1 )
      switch ( c ) {
	
      case 'N':
	N = (uint)atoi(optarg);
	break;

      case 'm':
	mode = atoi(optarg);
	break;

      case 'c':
	cutoff = (uint)atoi(optarg);
	break;

      case 'd':
	dispersion = (float)atof(optarg);
	break;
	
      case 's':
	seed = atol(optarg);
	break;	

      case 'f':
	int len = strlen(optarg)+1;
	filename = (char*)malloc( len + 1 );
	snprintf( filename, len, "%s", optarg );
	break;

      case 'h':
	printf("\npossible arguments: "
	       "-m _ -N _ -c _ -d _ -s _ -f _\n\n"
	       "\tm:   0 -> random values in the range [0: $cutoff]\n"
	       "\t     1 -> random values in the range [min : max ]\n"
	       "\t          min = $cutoff*(1-$f)\n"
	       "\t         mas = $cutoff*(1+$f)\n"
	       "\t     default is 0\n"
	       "\t-c:  the $cutoff value \n"
	       "\t     default is 500\n"
	       "\t-d:  the $f value\n"
	       "\t     default is 0.1"
	       "\t-N:  how many points\n"
	       "\t     default is 10000\n"
	       "\t-s:  the random seed\n"
	       "\t     default is 12345\n"
	       "\t-f:  the output file name\n"
	       "\t     default is \"nodes.dat\"\n\n" );
	exit(0);
	break;
	
      default:
	break;
	
      }


    if ( N == 0 ) {
      printf("generation of 0 elements done... it was easy\n");
      exit(0); }

    printf("generating %u values with seed %ld, output will be on file < %s >\n", N, seed, filename );
    if ( mode == 0 )
      printf("mode: random values in the range [0:%u]\n", cutoff);
    else
      printf("mode: random values in the range [%u:%u]\n",
	     (uint)(cutoff*(1 - dispersion)),
	     (uint)(cutoff*(1 + dispersion)) );
	

  }

  
  uint *nodes;
  FILE *file;    

  // check that we can create the file
  file = fopen( filename, "w" );
  if ( file == NULL )
    {
      printf("unable to open file %s\n", filename );
      exit(1);
    }
  // write how many items it will contain
  fwrite ( &mode, sizeof(int), 1, file );
  fwrite ( &N , sizeof(uint), 1, file );
  fwrite ( &cutoff, sizeof(uint), 1, file );
    
  // allocate the array that will contains the values
  // of the linked-list's nodes
  nodes = (uint*)malloc( sizeof(uint)* N );
  if ( nodes == NULL )
    {
      printf("not enough memory to allocate the "
	     "temporary array of values with %u entries\n", N );
      exit(2);
    }
  
  timing = CPU_TIME;
  if ( mode == 0 )
    {	    
      for ( uint i = 0; i < N; i++ )
	{
	  uint new = lrand48() % cutoff;
	  while ( new == 0 )
	    new = lrand48() % cutoff;
	  nodes[i] = new;
	}
    }  
  else
    {
      uint range = (uint)((float)cutoff * dispersion);
      fwrite ( &range, sizeof(uint), 1, file );
	
      for ( uint i = 0; i < N; i++ )
	nodes[i] = cutoff + mrand48() % range;
    }

  timing = CPU_TIME - timing;
  printf("took %g seconds\n", (double)timing/1e9 );

  
  fwrite( nodes, sizeof(uint), N, file );
  fclose(file);

    
  return 0;
}
