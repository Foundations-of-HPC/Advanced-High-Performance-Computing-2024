
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

#define DONT_USE_TASKYIELD  0
#define USE_TASKYIELD  1

typedef struct llnode
{
  int data;
  omp_lock_t lock;
 #if defined(DEBUG)
  int        owner;
 #endif
  
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
int       find_and_insert_parallel ( llnode_t *, int, int, _Atomic unsigned int *, llnode_t ** );


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
      omp_destroy_lock( &(head->lock) );
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


static inline int my_test_lock( llnode_t *node, int me )
{
  int got = omp_test_lock(&(node->lock));
  if ( got )
    SET_OWNER( node, me );
  return got;
}


static inline void my_unset_lock( llnode_t *node )
{
  UNSET_OWNER(node);
  omp_unset_lock(&(node->lock));
}



// ······················································


int find_and_insert_parallel( llnode_t *head, int value,
			      int use_taskyield,
			      _Atomic unsigned int *clashes,
			      llnode_t **new_node )
{
  
  if ( head == NULL )
    return -1;
  
  double timing = CPU_TIME_th;
  
  llnode_t *start = head;
  int done = 0;

  while ( !done )
    {

      // ------------------------------------------------------------------
      // STEP 1: Optimistic search (no locks held)
      // ------------------------------------------------------------------

      
      llnode_t *prev = NULL, *next = NULL;
      
      // find the first guess for the insertion point
      // NOTE: this look-up runs while other threads may be
      //       inserting nodes; atomic reads in find()
      //       prevent undefined bahviour, NOT possible
      //       inconsistencies!
      find ( start, value, &prev, &next );
  
      // at this point to our best knowledge:
      //    prev->data < value <= next->data
      // However, somebody may have inserted a new node
      // in between
  
      // ------------------------------------------------------------------
      // STEP 2: Acquire locks (deadlock-free protocol)
      // ------------------------------------------------------------------

      int locks_acquired = 0;
      while( !locks_acquired )
	{

	  // [ A ] acquire the lock on the prev first
	  if( prev != NULL )
	    {
	      // busy-wait for the lock
	      while ( omp_test_lock(&(prev->lock)) == 0 ) {	    
		if ( use_taskyield ) {
		 #pragma omp taskyield
		} } 

	      SET_OWNER(prev, me);
	      locks_acquired = 1;
	    }

	  
	  // [ B ] if next exists, acquire the lock
	  if ( next != NULL )
	    {
	      // we acquire the lock on the next
	      //
	      locks_acquired = my_test_lock(next, me);
	      if( !locks_acquired  )
		{
		  
		  if ( prev != NULL ) {
		    // let's release the prev and retry
		    my_unset_lock(prev); }
		  
		  if ( use_taskyield ) {
		    // if we wnat to use the taskyield feature,
		    // that is a good moment ?
		   #pragma omp taskyield
		  }
		}
	      else
		SET_OWNER(next, me);
	    }      
	}


      // ------------------------------------------------------------------
      // STEP 3: check that insertion point is still valid
      // ------------------------------------------------------------------
      // While we were acquiring locks, another task might have inserted
      // a node between prev and next. Check for this:
      
      llnode_t *prev_next = (prev != NULL) ? 
	atomic_load_explicit(&prev->next, memory_order_relaxed) : NULL;
      llnode_t *next_prev = (next != NULL) ? 
	atomic_load_explicit(&next->prev, memory_order_relaxed) : NULL;      
      
      // meanwhile, did somebody already insert a node between prev and next?
      if( ( (prev != NULL) && (prev_next != next) ) ||
	  ( (next != NULL) && (next_prev != prev) ) )
	{

	  // Count the retry (for statistics)
	  atomic_fetch_add_explicit(clashes, 1, memory_order_relaxed);

	  // we want to repeat the find of the insertion point guess
	  // from an updated pointer
	  start = ( (prev!=NULL) ? prev : next );

	  // release the locks
	  if ( prev != NULL )
	    my_unset_lock(prev);
	  
	  if ( next != NULL )
	    my_unset_lock(next);

	  continue;
	}


      // ------------------------------------------------------------------
      // STEP 4: Perform insertion (locks held, validation passed)
      // ------------------------------------------------------------------

      // at this point, we have found our new left and right pointers,
      // and we have their locks  
      
      // allocate a new node
      // HINT: transform everything in an array of nodes
      //
      llnode_t *new = (llnode_t*)malloc( sizeof(llnode_t) );
      *new_node = new;
      if ( new == NULL )
	return -2;
      
      // initialize the new node
      // omp_init_lock_with_hint( &(new->lock), omp_lock_hint_contended );
      omp_init_lock( &(new->lock) );
      SET_OWNER(new, me);
      new->data = value;

      // Link the new node into the list
      // These atomic stores are released by the subsequent lock releases
      atomic_store_explicit(&new->prev, prev, memory_order_relaxed);
      atomic_store_explicit(&new->next, next, memory_order_relaxed);

      if ( prev != NULL )
	atomic_store_explicit(&prev->next, new, memory_order_relaxed);
      if ( next != NULL)
	atomic_store_explicit(&next->prev, new, memory_order_relaxed);

      // ------------------------------------------------------------------
      // STEP 5: Release all locks
      // ------------------------------------------------------------------
      // NOTE: Lock releases provide memory_order_release semantics
      // Subsequent lock acquires will see our writes

      if ( prev != NULL )
	my_unset_lock(prev);
      
      if( next != NULL )
	my_unset_lock(next);
      
      done = 1;

    }  // closes main while ( !done )

  timing = CPU_TIME_th - timing;
  insertiontime += timing;
  return 0;
}


// ······················································

int main ( int argc, char **argv )
{
  int N, mode;
  long int seed;
  {
    int a = 1;
    N    = ( argc > 1 ? atoi(*(argv+a++)) : 1000000 ); 
    mode = ( argc > a ? atoi(*(argv+a++)) : DONT_USE_TASKYIELD );
    seed = ( argc > a ? atoi(*(argv+a++)) : 98765 );

    if ( seed == 0 )
      seed = time(NULL);
    srand48( seed );
  }


  llnode_t *head = (llnode_t*)malloc(sizeof(llnode_t));
  head->data = lrand48();
  atomic_store_explicit(&head->prev, NULL, memory_order_relaxed);
  atomic_store_explicit(&head->next, NULL, memory_order_relaxed);

  //omp_init_lock_with_hint( &(head->lock), omp_lock_hint_contended );
  // if the compiler does not support lock_with_hint,
  // use the unhinted initialization
  omp_init_lock( &(head->lock) );

  
  ull timing = CPU_TIME;
  int norm   = N*N;
  
  _Atomic unsigned int clashes = 0;
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
		find_and_insert_parallel( head, new_value, mode, &clashes, &head );
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

  char string[23] = {0};
  sprintf( string, " with %u clashes", clashes);  
  printf("generation took %g seconds (wtime) %s\n", ((double)timing/1e9), string);

  printf("verifying.. "); fflush(stdout);
    
  head = get_head( head );

  int actual_nodes = walk( head, DEBUG_SET );
  if ( actual_nodes != N )
    printf("\nshame on me! %d nodes instaed of %d have been found!",
	   actual_nodes, N);
  else
    printf("ok\n");

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
  
  // cleanup
  printf("cleaning up..\n"); fflush(stdout);
  delete_all ( head );


  
  
  return 0;
}
