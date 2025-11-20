
#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#if !defined(_OPENMP)
#error "please switch on the OpenMP support to compile this source file"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <stdatomic.h>  // C11 atomics


// =========================================================================
//
//  define useful quantities
//

typedef unsigned long long ull;
#define TIME_CUT 100000000


int me;
unsigned long long int tasktime = 0, insertiontime = 0;
#pragma omp threadprivate(me, tasktime, insertiontime)

#define CPU_TIME ({ struct timespec ts; (clock_gettime( CLOCK_TAI, &ts ), \
					 (ull)ts.tv_sec * 1000000000 +	\
					 (ull)ts.tv_nsec); })

#define CPU_TIME_th ({ struct  timespec ts; (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &ts ), \
					     (ull)ts.tv_sec*1000000000 + \
					     (ull)ts.tv_nsec); })

#if defined(DEBUG)

#define DEBUG_SET 1
#define TIMESTAMP (CPU_TIME % TIME_CUT)
#define dbgout(...) printf( __VA_ARGS__ );
#define SET_OWNER( ptr, me ) ((ptr)->owner = (me))
#define UNSET_OWNER( ptr ) ((ptr)->owner = -1)
#else

#define DEBUG_SET 0
#define TIMESTAMP
#define dbgout(...)
#define SET_OWNER( ptr, own )
#define UNSET_OWNER( ptr )
#endif


//
// =========================================================================
//
// define data structures
//

typedef struct llnode
{
  int data;
  
  _Atomic(struct llnode *)next;
  _Atomic(struct llnode *)prev;  
} llnode_t;


//
// =========================================================================
//
// prototypes
//

llnode_t* get_head        ( llnode_t *);
int       walk            ( llnode_t *, int);
int       delete_all      ( llnode_t * );
int       find            ( llnode_t *, int, llnode_t **, llnode_t ** );
int       find_and_insert ( llnode_t *, int, llnode_t ** );


//
// =========================================================================
// =========================================================================


// ······················································

llnode_t *get_head ( llnode_t *start )
/*
 * serially walk the list basck to find
 * the list head
 * returns the head
 */
{
  llnode_t *ptr = start;
  llnode_t *prev_node = atomic_load_explicit(&start->prev, memory_order_relaxed);
  while( prev_node != NULL )
    {
      ptr = prev_node;
      prev_node = atomic_load_explicit(&ptr->prev, memory_order_relaxed);
    }
  
  return ptr;
}

// ······················································

int walk ( llnode_t *start, int mode )
/*
 * serial walk the list starting from the node start
 * as first, the list is walked back until the list head
 * if mode == 1, the list is then walked ahed printing
 * the first 100 nodes.
 */
{
  if ( start == NULL )
    return -1;
  
  if(mode) printf("%9d [-]", start->data );
  
  int n = 1;
  int prev_value = start->data;
  
  llnode_t *next_node = atomic_load_explicit(&start->next, memory_order_relaxed);
  while( next_node != NULL)
    {
      int data = next_node->data;
      
      if( mode )
	{
	  if (n < 100 )
	    printf( "%9d %s ",
		    data,
		    (data < prev_value? "[!]":"[ok]") );
	  else if ( n == 100)
	    printf( "..." );
	  prev_value = data;
	}
      
      n++;
      next_node = atomic_load_explicit(&next_node->next, memory_order_relaxed);
    }
  return n;
}


// ······················································

int delete_all( llnode_t *head )
/*
 * delete all the nodes
 * destroy every lock
 */
{
  while ( head != NULL )
    {
      llnode_t *next_node = atomic_load_explicit(&head->next, memory_order_relaxed);
      free( head );
      head = next_node;      
    }
  return 0;
}


// ······················································

// find the insertion point for value, starting from head
// *prev and *next will point to the left and right nodes
// either one being null if we arrived at the head or at
// the tail of the linked list
//
// // INTENTIONAL DATA RACE: We read list nodes concurrently with writes
// by other tasks. This is technically undefined behavior in C11/C17.
// In practice, on x86-64 with naturally aligned pointers:
// - Pointer reads are atomic at the hardware level
// - We may see stale data, but that's acceptable (find() just provides
//   a search hint)
// - DO NOT use with TSAN or thread sanitizers (will report false positives)
//
// This demonstrates that sometimes practical concurrency tolerates
// races that are formally undefined.
//
int find ( llnode_t *head, int value, llnode_t **prev, llnode_t **next )
{
  *prev = NULL, *next = NULL;
  
  if ( head == NULL )
    // The first node must exist in this simple
    // implementation.
    // To improve that, pass **head instead
    // of *head
    return -1;

  int       nsteps = 0;

  if ( head->data > value )
    {
      // ----------------------------------------------------------------
      // WALK BACKWARDS to find insertion point
      // ----------------------------------------------------------------

      // <*prev> and <*next> may be changing;
      // we need atomic read to formally avoid data race
      //
      llnode_t *ptr = atomic_load_explicit(&head->prev, memory_order_relaxed);
      *next = head;

      // Walk backwards while data > value
      //
      while ( ptr != NULL )
	{
	  // note: the <data> field is not changing;
	  // simple read is ok
	  int this_data = ptr->data;

	  if ( this_data <= value )
	    break;

	  *next = ptr;
	  ptr = atomic_load_explicit(&ptr->prev, memory_order_relaxed);
	  nsteps++;
	}
      *prev = ptr;
    }
  else
    {
      // ----------------------------------------------------------------
      // WALK FORWARDS to find insertion point
      // ----------------------------------------------------------------

      // <*prev> and <*next> may be changing;
      // we need atomic read to formally avoid data race
      //
      llnode_t *ptr = atomic_load_explicit(&head->next, memory_order_relaxed);
      *prev = head;

      // Walk forwards while data < value
      while ( ptr != NULL )
	{
	  int this_data = ptr->data;
	  
	  if ( this_data >= value )
	    break;  // Found it: (*prev)->data < value <= ptr->data
	  
	  *prev = ptr;
	  ptr = atomic_load_explicit(&ptr->next, memory_order_relaxed);
	  nsteps++;
	}      

      *next = ptr;
    }

  return nsteps;
}


// ······················································


int find_and_insert( llnode_t *head, int value,
		     llnode_t **new_node )
{
  
  if ( head == NULL )
    return -1;


  double timing = CPU_TIME_th;
  
  // ------------------------------------------------------------------
  // STEP 1: find the insertion point
  // ------------------------------------------------------------------
  
  
  llnode_t *prev = NULL, *next = NULL;
  
  // find the first guess for the insertion point
  // NOTE: this look-up runs withou concurrency
  //       since the execution of find is locked
  //
  find ( head, value, &prev, &next );
  


  // ------------------------------------------------------------------
  // STEP 2: Perform insertion 
  // ------------------------------------------------------------------


  // NOTE: we stiil use atomic store just to bridge to the next step
  // they are not really needed here
      
  // allocate a new node
  // HINT: transform everything in an array of nodes
  //
  llnode_t *new = (llnode_t*)malloc( sizeof(llnode_t) );
  *new_node = new;
  if ( new == NULL )
    return -2;
  
  // initialize the new node

  new->data = value;

  // Link the new node into the list

  atomic_store_explicit(&new->prev, prev, memory_order_relaxed);
  atomic_store_explicit(&new->next, next, memory_order_relaxed);
  
  if ( prev != NULL )
    atomic_store_explicit(&prev->next, new, memory_order_relaxed);
  if ( next != NULL)
    atomic_store_explicit(&next->prev, new, memory_order_relaxed);

  timing = CPU_TIME_th - timing;
  insertiontime += timing;
  return 0;
}


// ······················································

int main ( int argc, char **argv )
{
  int N;
  long int seed;
  {
    int a = 1;
    N    = ( argc > 1 ? atoi(*(argv+a++)) : 1000000 ); 
    seed = ( argc > a ? atoi(*(argv+a++)) : 98765 );

    if ( seed == 0 )
      seed = time(NULL);
    srand48( seed );
  }


  omp_lock_t insertion_lock;
  omp_init_lock( &insertion_lock );
  
  
  llnode_t *head = (llnode_t*)malloc(sizeof(llnode_t));
  head->data = lrand48();
  atomic_store_explicit(&head->prev, NULL, memory_order_relaxed);
  atomic_store_explicit(&head->next, NULL, memory_order_relaxed);

  ull timing = CPU_TIME;
  int norm   = N*N;
  
 #pragma omp parallel
  {
    me = omp_get_thread_num();

   #pragma omp single
    {
      printf("running with %d threads, thread %d generates tasks\n",
	     omp_get_num_threads(), me);

      int nsteps   = 10;
      int step     = (N/10);
      int nextstep = 1;

      int N_1      = N-1;
      int n        = 0;

      while ( n < N_1 )
	{
	 #define BATCH_SIZE 100
	  int this_batch_size = ( BATCH_SIZE > N-n-1 ? N-n-1 : BATCH_SIZE );

	 #pragma omp task firstprivate(n, head, this_batch_size)
	  {
	    double timing = CPU_TIME_th;
	    unsigned short int seeds[3] = {(seed^me)^n, seed+me-n, me*n};

	    for ( int batch = 0; batch < this_batch_size; batch++ )
	      {
		int new_value = nrand48(seeds) % norm;
		omp_set_lock( &insertion_lock );
		find_and_insert( head, new_value, &head );
		omp_unset_lock( &insertion_lock );
	      }
	    timing = CPU_TIME_th - timing;
	    tasktime += timing;
	  }
	  n += this_batch_size;
	  if ( n >= (step*nextstep)  ) {
	    printf("%.1f%% of nodes pinned to tasks\n", (double)n/N*100);
	    nextstep++; }
	}
    }

  }

  timing = CPU_TIME - timing;

  omp_destroy_lock( &insertion_lock );

  printf("generation took %g seconds (wtime)\n", ((double)timing/1e9));
  printf("verifying.. "); fflush(stdout);
    
  head = get_head( head );

  int actual_nodes = walk( head, DEBUG_SET );
  if ( actual_nodes != N )
    printf("\nshame on me! %d nodes instaed of %d have been found!",
	   actual_nodes, N);
  else
    printf("ok\n");


  // ========================================================================
  //  timing statistics


  unsigned long long nthreads;
  unsigned long long task_avg = 0, task_min = tasktime, task_max = tasktime;
  unsigned long long  ins_avg = 0, ins_min = insertiontime, ins_max = insertiontime;
  double  ratio_avg = 0, ratio_min = (double)insertiontime/tasktime, ratio_max = (double)insertiontime/tasktime;
 #pragma omp parallel reduction(+:task_avg,ins_avg,ratio_avg) reduction(min:task_min, ins_min, ratio_min) reduction(max:task_max,ins_max, ratio_max)
  {
   #pragma omp single
    nthreads = (unsigned long long)omp_get_num_threads();
    
    task_avg += tasktime;
    ins_avg += insertiontime;
    
    task_min = tasktime;
    task_max = tasktime;
    
    ins_min  = insertiontime;
    ins_max  = insertiontime;
    
    double ratio = (double)insertiontime / (double)tasktime;
    ratio_avg += ratio;
    ratio_min = ratio;
    ratio_max = ratio;
  }

  printf("avg, min and max task time [ns]: %g %llu %llu\n"
	 "avg, min and max ins  time [ns]: %g %llu %llu\n"
	 "-- avg, min and max ins/task  time : %g %g %g\n",
	 (double)task_avg/nthreads, task_min, task_max,
	 (double)ins_avg/nthreads, ins_min, ins_max,
	 ratio_avg / nthreads, ratio_min, ratio_max );

  // ========================================================================
  
  // cleanup
  printf("cleaning up..\n"); fflush(stdout);
  delete_all ( head );

    return 0;
}
