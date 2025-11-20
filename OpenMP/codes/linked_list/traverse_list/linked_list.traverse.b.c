
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


// ··········································.
//  defines values for the
//  case to run
//
#define FIRST_CASE     0
#define FOR_STATIC     FIRST_CASE
#define FOR_DYNAMIC    1
#define USE_TASKS      2
#define LAST_CASE      3

char *case_names[LAST_CASE] =  {"running static loop over array"
				"running dynamic loop over array"
				"running with tasks"};

// ··········································.
//  defines the values for the various
//  options in the walk routines
//
#define WALK_CHECK              0           // walk the list and check the sorting
#define WALK_COUNT_ACTIVENODES  1           // count the active nodes
#define WALK_INSERT_ACTIVENODES 2           // insert active nodes into an array
#define WALK_GENERATE_TASKS     3           // generate tasks to process active nodes
#define WALK_GENERATE_TASKSGROUPS 4
#define WALK_VERIFY 5                       // verify that all active nodes have been
					    // correctly processed

//
// =========================================================================
//
// define data structures
//


typedef struct llnode
{
  uint data;
  uint result;
  struct llnode *next;
  struct llnode *prev;  
} llnode_t;



//// =========================================================================
//
// declare global data

uint  howmanynodes= 0;                       // how many nodes have been processed, fir every thread
ull   howmuchtime;                           // how much time has been spent in processing nodes, per thread
#pragma omp threadprivate(howmanynodes, howmuchtime)

char  filename_dflt[] = "nodes.dat";
uint  ActiveThreshold;
uint  MaxLimit;

//
// =========================================================================
//
// prototypes
//


inline llnode_t *create_node         ( uint, llnode_t *, llnode_t * );
       uint      load_nodes_and_build_list  ( llnode_t **, char*, int *, uint *, uint * );
       uint      find                ( llnode_t *, uint, llnode_t ** restrict, llnode_t ** restrict );
       uint      find_and_insert     ( llnode_t **, uint );

inline int       is_active           ( uint );
       void      process_nodes_group ( llnode_t * restrict * restrict, uint);
       ull       process_node        ( llnode_t * restrict, uint );

       llnode_t *get_head            ( llnode_t * restrict );
       int       delete              ( llnode_t * restrict );
       uint      walk                ( llnode_t * restrict, int, void * restrict );


//
// =========================================================================
// =========================================================================


int is_active( uint data  )
{
  return (data >= ActiveThreshold);
}

void process_nodes_group( llnode_t * restrict list[], uint maxlimit )
{
  ull timing = CPU_TIME_th;
  // ·························
  
  int idx    = 0;  
  while ( list[idx] != NULL )
    {
      llnode_t *node  = (llnode_t*)list[idx];
      ull       wow   = 0;
      ull       limit = node->data;
      limit = (limit > maxlimit ? maxlimit : limit);
 
      for ( uint i = 0; i < limit ; i++ )
	wow += i;;
      ((llnode_t*)node)->result = wow;
      howmanynodes++;

      idx++;
    }

  // ·························
  howmuchtime += CPU_TIME_th - timing;
  return;
}


ull process_node( llnode_t * restrict node, uint maxlimit )
{
  ull timing = CPU_TIME_th;
  // ·························

  ull wow   = 0;
  ull limit = node->data;
  limit = (limit > maxlimit ? maxlimit : limit);
  
  for ( uint i = 0; i < limit ; i++ )
    wow += i;
  ((llnode_t*)node)->result = wow;
  howmanynodes++;
  
  // ·························
  howmuchtime += CPU_TIME_th - timing;  
  return wow;
}


// ······················································

llnode_t *get_head ( llnode_t *start )
/*
 * walk the list basck to find the list head
 * returns the head
 */
{
  llnode_t *walk = start;
  while( walk->prev != NULL )
    walk = walk->prev;
  
  return walk;
}


// ······················································

int delete ( llnode_t *head )
/*
 * delete all the nodes
 */
{
  head = get_head(head);
  while ( head != NULL )
    {
      llnode_t *prev = head;
      head = head->next;
      free( prev );
    }
  return 0;
}

// ······················································

uint walk ( llnode_t *start, int mode, void *pointer )
/*
 * walk the list starting from the node start
 * as first, the list is walked back until the list head
 * if mode == 1, the list is then walked ahed printing
 * the first 100 nodes.
 */
{
  uint ncount  = 0;
  uint _count_ = 0;

  void **array = (void**)pointer;
  uint  *count = (uint*)pointer;
  uint   chunk_size;

  if ( mode == WALK_GENERATE_TASKSGROUPS )
    chunk_size = *(uint*)pointer;
  else
    chunk_size = 1;
  
  if ( start != NULL )
    {
      
      llnode_t *walk = start;
      llnode_t *list[chunk_size+1];
      for ( int i = 0; i <= chunk_size; i++ ) list[i] = NULL;
      
      int  list_size = 0;
      uint prev_value = (walk != NULL ? walk->data : 0);
      
      while( walk != NULL)
	{
	  switch ( mode )
	    {

	      // check that the nodes are sorted
	      //
	    case WALK_CHECK : {
	      _count_++;
	      ncount += (walk->data < prev_value);
	      prev_value = walk->data; } break;
	      
	      // count how many nodes have to be processed
	      //
	    case WALK_COUNT_ACTIVENODES: 
	      ncount += is_active ( walk->data );
	      break;
	      
	      // insert active nodes into an array
	      //
	    case WALK_INSERT_ACTIVENODES: {
	      if ( is_active ( walk->data ) )
		array[ncount++] = (void*)walk; }
	      break;

	      // generate a task per every node
	      //
	    case WALK_GENERATE_TASKS: {
	      if ( is_active ( walk->data ) )
		ncount++;
	     #pragma omp task firstprivate(walk) untied
	      process_node( walk, MaxLimit ); };
	      break;

	      // generate a task every TASKGROUP_SIZE nodes
	      //
	    case WALK_GENERATE_TASKSGROUPS: {
	      if ( is_active ( walk->data ) )
		{
		  ncount++;
		  list[list_size++] = walk;
		  if ( list_size == chunk_size ) {
		   #pragma omp task firstprivate(walk, list) untied
		    process_nodes_group( list, MaxLimit );
		    memset( list, 0, sizeof(void*)*chunk_size);
		    list_size = 0; }
		} }
	      break;

	      // verify that all the active nodes have been processed
	      // and that the results are correct
	    case WALK_VERIFY: {
	      // count how many active nodes have been processed
	      _count_ += (walk->result > 0); 
	      if ( is_active( walk->data ) )
		{		  
		  ull wow   = walk->data * (walk->data-1) / 2;
		  ncount += (wow != walk->result);
		} }
	      break;
	      
	    }
	  
	  walk = walk->next;
	}
      
      if ( (mode == WALK_GENERATE_TASKSGROUPS) && (list_size > 0 ) )
       #pragma omp task firstprivate(walk, list) untied
	process_nodes_group( list, MaxLimit );

    }
      
  if ( ((mode == WALK_CHECK)||(mode == WALK_VERIFY)) && (count != NULL ) )
    *count = _count_;
  return ncount;
}



// ······················································

uint find ( llnode_t *head, uint value, llnode_t **prev, llnode_t **next )
{
  *prev = NULL, *next = NULL;
  
  if ( head == NULL )
    // The first node must exist in this simple
    // implementation.
    // To improve that, pass **head instead
    // of *head
    return -1;

  uint       nsteps = 0;
  llnode_t  *ptr = NULL;

  if ( head-> data > value )
    {
      // we need to walk back
      //
      ptr  = head->prev;
      *next = head;
      while ( (ptr != NULL) && (ptr->data > value) )
	{
	  *next = ptr;
	  ptr  = ptr->prev;
	  nsteps++;
	}
      *prev = ptr;
    }
  else
    {
      // we need to walk ahead
      //
      ptr  = head->next;
      *prev = head;
      while ( (ptr != NULL) && (ptr->data < value) )
	{
	  *prev = ptr;
	  ptr  = ptr->next;
	  nsteps++;
	}
      *next = ptr;
    }

  return nsteps;
}


// ······················································
llnode_t * create_node ( uint data, llnode_t *prev, llnode_t *next )
{
  llnode_t *node = (llnode_t*)malloc( sizeof(llnode_t) );
  if ( node != NULL )
    {
      node-> data   = data;
      node-> result = 0;
      node-> prev   = prev;
      node-> next   = next;
    }
  return node;
}


// ······················································
uint find_and_insert( llnode_t **head, uint value )
{
  if ( *head == NULL )
    {
      *head = create_node( value, NULL, NULL );
      if ( *head == NULL )
	return -2;
    }
  else
    {
      llnode_t *prev = NULL, *next = NULL;

      find ( *head, value, &prev, &next );
  
      llnode_t *new = create_node( value, prev, next);

      if ( new != NULL )
	{
	  if( prev != NULL )
	    prev->next = new;
	  if( next != NULL )
	    next->prev = new;
	}
      else
	return -2;
    }
  
  return 0;
}


uint load_nodes ( char *filename,
		  int *mode, uint *cutoff, uint *range,
		  uint **nodes)
{

  FILE *file = fopen( filename, "r" );
  if ( file == NULL ) {
    printf ( "unable to open the file %s\n", filename );
    *mode = 1;
    return 0; }

  uint    N;
  size_t  ret;
  
  // ···········································
  // load data from the file
  // 
  ret = fread ( mode, sizeof(uint), 1, file );
  ret = fread ( &N, sizeof(uint), 1, file );
  ret = fread ( cutoff, sizeof(uint), 1, file );
  if ( *mode == 1 )
    ret = fread( range, sizeof(uint), 1, file );
  else
    *range = 0;

  *nodes = (uint*) malloc( sizeof(uint)*N );
  ret    = fread ( *nodes, sizeof(uint), N, file );
  fclose(file);
  //
  // ···········································
  
  return N;
}
  




/*

   ············································································································
   ············································································································
   ············································································································

*/


int main ( int argc, char **argv )
{
  
  // options controlling how the nodes are generated
  //
  uint  N;               // how many nodes to generate
    
  long int seed;         // if needed
  
  
  // options controlling the case to run
  int   case_to_run;     // 1: static for-loop over an array
                         // 2: dynamic for-loop over array
                         // 3: tasks
                         
  int   chunk;           // more details on the case to run
			 // for loop: the size of the chunks for the scheduling,
			 //           0 means do not specify a value
			 // tasks:    the groupsize of the task (i.e. how many
			 //           nodes are included in a task)

  float fraction;
  
  char     *filename;
  llnode_t *head = NULL;
  ull       timing;
  

  {
    // parse the command line arguments
    // let's use getopt since we use a simple command line
  
    // set-up the default
    N              = 0;
    case_to_run    = 0;
    chunk          = 1;
    MaxLimit       = 0;
    fraction       = 0.6;
    filename       = filename_dflt;

    int const_value = 0;
    int c;
    
    while ( (c = getopt(argc, argv, "N:c:f:C:s:F:v:h"))  != -1 )
      switch ( c ) {

      case 'N':
	N = atoi(optarg);
	break;
	
      case 'c':
	case_to_run = atoi(optarg);
	break;

      case 'f':
	int len = strlen(optarg);
	filename = (char*)malloc( len+2 );
	snprintf( filename, len+1, "%s", optarg );
	break;

      case 'M':
	MaxLimit = (uint)atoi( optarg );
	break;
	
      case 'v':
	const_value = (uint)atoi( optarg );
	break;
	
      case 'C':
	chunk = (uint)atoi(optarg);
	break;

      case 'F':
	fraction = (float)atof(optarg);
	break;

      case 's':
	seed = atoi(optarg);
	break;

      case 'h':
	printf("\npossible arguments: "
	       "-c _ -C _ -F _ -M _ -f _ -v _ -N _ -s _\n\n"
	       "\tc :  0 -> for loop with static schedule\n"
	       "\t     1 -> for loop with dynamic schedule\n"
	       "\t     2 -> tasks\n"
	       "\tC :  the chunk size for case 0,1 or the bunch size for case 2\n"
	       "\tF :  the fraction of nodes that will be \"active\"\n"
	       "\tM :  the max value to be used when processing a node\n"
	       "\tf :  the name of the file that contains the nodes' values\n"
	       "\tv :  the value to be used to generate a nu,ber of nodes with constant value\n"
	       "\tN :  the number of nodes to be generated with constant value\n" );
	exit( 0 );
	break;
	
      default:
	break;
	
      }

    if ( (case_to_run < FIRST_CASE) || (case_to_run >= LAST_CASE) ) {
      printf ( "unknown case %d\n", case_to_run );
      exit(1); }

    if ( ( (const_value != 0) ||
	   (N > 0) ) &&
	 (const_value*N == 0) )
      {
	printf ("if you want to generate nodes with constant values "
		"you must specify both the values using -N and -v options\n");
	return -1;
      }
    
    if ( (const_value > 0)  &&
	 (filename != filename_dflt) )
      {
	printf ( " you can not require to load data from file %s "
		 "and at the same time to generate %u nodes "
		 "with constant value %u\n",
		 filename, N, const_value );
	return -2;
      }
    
    if ( const_value == 0 )
      {
	uint *nodes = NULL;
	int   mode;
	uint  cutoff;
	uint  range;

	printf ( " > loading the values.. " );
	timing = CPU_TIME;
	N = load_nodes( filename, &mode, &cutoff, &range, &nodes );
	timing = CPU_TIME - timing;
	printf ( "took %g seconds\n", timing/1e9);
	
	if ( N == 0 )
	  exit(mode);
    
	if ( mode == 0 ) {
	  ActiveThreshold = (1-fraction)*cutoff;
	  MaxLimit = (MaxLimit == 0 ? cutoff : MaxLimit); }
	else if ( mode == 1 )  {	  
	  ActiveThreshold = cutoff-range + 2*range*(1-fraction);
	  MaxLimit = (MaxLimit == 0 ? cutoff : cutoff+range); }
	  
	
	printf ( " > building the nodes.. " );
	timing = CPU_TIME;
	uint i_10 = N/10;
	uint i;
	for ( i = 0; i < N; i++ )
	  {
	    int res = find_and_insert( &head, nodes[i] );
	    if ( res < 0 )
	      break;
	    if ( i % i_10 == 0 )
	      printf("%u%%.. ", 10*i/i_10); fflush(stdout);
	  }
	timing = CPU_TIME - timing;
	printf ( "took %g seconds\n", timing/1e9);	
	free ( nodes );
	if ( i < N )
	  {
	    printf("unable to allocate room for "
		   "%u nodes (limit was at %u)\n",
		   N, i );
	    delete ( head );
	    exit( -1 );
	  }
      }
    else
      {
	ActiveThreshold = const_value*0.99;
	if ( MaxLimit == 0 )
	  MaxLimit = const_value;

	uint i;
	for ( i = 0; i < N; i++ )
	  {
	    int res = find_and_insert( &head, const_value );
	    if ( res < 0 )
	      break;
	  }
	if ( i < N )
	  {
	    printf("unable to allocate room for "
		   "%u nodes (limit was at %u)\n",
		   N, i );
	    delete ( head );
	    exit( -1 );
	  }	
      }

    head = get_head( head );
  }

  
  
  printf(" > checking the list.. "); fflush(stdout);
  timing = CPU_TIME;

  uint actual_nodes;
  uint faults = walk( head, WALK_CHECK, &actual_nodes);

  timing = CPU_TIME - timing;
  printf ( "took %g seconds\n", (double)timing/1e9 );  

  if ( actual_nodes != N )
    printf("shame on me! %u nodes instead of %u have been found!",
	   actual_nodes, N);
  if ( faults > 0 )
    printf("a tragedy occured.. %u nodes are unsorted!!\n", faults );

  printf(" > counting active nodes.. "); fflush(stdout);
  timing = CPU_TIME;
  
  uint Nactive = walk( head, WALK_COUNT_ACTIVENODES, NULL );

  timing = CPU_TIME - timing;
  printf ( "%u active nodes found (took %g seconds)\n", Nactive, (double)timing/1e9 );

  if ( Nactive <= 0 )
    {
      delete ( head );
      exit(0);
    }

  printf(" >>  now processing\n");
  ull alltime = CPU_TIME;
   
  if ( case_to_run < USE_TASKS )
    {
      // build an array of pointers to nodes and process it using an omp for loop

      void **array;
      array = (void**)calloc( Nactive, sizeof(void*) );
      if ( array == NULL )
	{
	  printf("not enough memory to host %u pointers to active nodes\n", Nactive );
	  delete(head);
	  exit(1);
	}

      printf( "\t * preparing the array.. "); fflush(stdout);
      timing = CPU_TIME;
      walk( head, WALK_INSERT_ACTIVENODES, (void*)array );
      timing = CPU_TIME - timing;
      printf ( "took %g seconds\n", (double)timing/1e9 );
      
      printf ( "\t * processing the active nodes.. "); fflush(stdout);

      timing = CPU_TIME;
      if ( case_to_run == FOR_STATIC )
	{
	 #pragma omp parallel for schedule(static, chunk)
	  for ( int i = 0; i < Nactive; i++ )
	    process_node( array[i], MaxLimit );

	}
      else if ( case_to_run == FOR_DYNAMIC )
	{
	 #pragma omp parallel for schedule(dynamic, chunk)
	  for ( int i = 0; i < Nactive; i++ )
	    process_node( array[i], MaxLimit );	      
	}
      timing = CPU_TIME - timing;
      printf ( "took %g seconds\n", (double)timing/1e9 );

      free ( array );
    }

  else 
    {
      // use tasks
      
      printf( "\t * generating the tasks and processing the array.. "); fflush(stdout);
      timing = CPU_TIME;
     #pragma omp parallel
      {
	int mode = (chunk > 1 ? WALK_GENERATE_TASKSGROUPS : WALK_GENERATE_TASKS );
       #pragma omp single
	{
	 #pragma omp task untied
	  walk( head, mode, &chunk );
	}
      }
      timing = CPU_TIME - timing;
      printf ( "took %g seconds\n", (double)timing/1e9 );
      
    }

  alltime = CPU_TIME - alltime;
  printf ( "\ntotal time: %g seconds\n", (double)alltime / 1e9 );
  
 #pragma omp parallel
  {
    int nthreads = omp_get_num_threads();
   #pragma omp for ordered
    for ( int i = 0; i < nthreads; i++ )
     #pragma omp ordered
      printf(" - thread %d processed %u nodes in %g seconds\n", i, howmanynodes, howmuchtime/1e9 );
    
    //free(myresults);
  }
  
  uint verified;
  faults = walk( head, WALK_VERIFY, &verified );
  if ( faults )
    printf( "%u faults found!\n", faults );
  if ( Nactive != verified )
    printf( "%u verified instead of %u\n", verified, Nactive );
  
  delete ( head );

  
 #undef USE_ARRAY
  return 0;
}
