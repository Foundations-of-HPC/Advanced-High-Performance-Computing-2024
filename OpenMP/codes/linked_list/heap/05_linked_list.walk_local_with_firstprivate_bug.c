
#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>


// =========================================================================
//
//  define useful quantities
//

typedef unsigned long long ull;
#define TIME_CUT 1000000009


#if defined(_OPENMP)

int me;
#pragma omp threadprivate(me)

#define CPU_TIME ({ struct timespec ts; (clock_gettime( CLOCK_TAI, &ts ), \
					 (ull)ts.tv_sec * 1000000000 +	\
					 (ull)ts.tv_nsec); })

#define CPU_TIME_th ({ struct  timespec ts; (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), \
					     (ull)myts.tv_sec*1000000000 + \
					     (ull)myts.tv_nsec); })

#else

#define CPU_TIME ({ struct timespec ts; (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), \
					 (ull)ts.tv_sec * 1000000000 +	\
					 (ull)ts.tv_nsec); })
#endif


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
 #if defined(_OPENMP)
  omp_lock_t lock;
 #if defined(DEBUG)
  int        owner;
 #endif
 #endif
  
  struct llnode *next;
  struct llnode *prev;  
} llnode_t;


//
// =========================================================================
//
// prototypes
//

llnode_t* get_head        ( llnode_t *);
int       walk            ( llnode_t *, int );
int       delete_all      ( llnode_t * );
int       find            ( llnode_t *, int, llnode_t **, llnode_t ** );
int       find_and_insert ( llnode_t *, int );

#if defined(_OPENMP)
int       find_and_insert_parallel ( llnode_t *, int, int, unsigned int *, llnode_t ** );
#endif

//
// =========================================================================
// =========================================================================


// ······················································

llnode_t *get_head ( llnode_t *start )
/*
 * walk the list basck to find the list head
 * returns the head
 */
{
  while( start->prev != NULL )
    start = start->prev;
  
  return start;
}

// ······················································

int walk ( llnode_t *start, int mode )
/*
 * walk the list starting from the node start
 * if mode == 1, the list is then walked ahed printing
 * the first 100 nodes.
 */
{
  if ( start == NULL )
    return -1;

  int n = 1;
  int prev_value = start->data;
  if(mode) printf("%9d [-]", start->data );
  
  start = start->next;
  while( start != NULL)
    {
      
      if( mode )
	{
	  if (n < 100 )
	    printf( "%9d %s ",
		    start->data,
		    (start->data < prev_value? "[!]":"[ok]") );
	  else if ( n == 100)
	    printf( "..." );
	  prev_value = start->data;
	}
      
      n++;
      start = start->next;
    }
    
  printf("\n");
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
      llnode_t *prev = head;
      head = head->next;
     #if defined(_OPENMP)
      omp_destroy_lock( &(prev->lock) );
     #endif
      free( prev );
    }
  return 0;
}


// ······················································

// find the insertion point for value, starting from head
// *prev and *next will point to the left and right nodes
// either one being null if we arrived at the head or at
// the tail of the linked list
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
  llnode_t *ptr = NULL;

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

static inline int my_test_lock( llnode_t *node, int me )
{
  int got = omp_test_lock(&(node->lock));
  if ( got )
    SET_OWNER(node, me);

  return got;
}


static inline void my_unset_lock( llnode_t *node )
{
  UNSET_OWNER(node);
  omp_unset_lock(&(node->lock));
}

int find_and_insert( llnode_t *head, int value )
{
  if ( head == NULL )
    // The first node must exist in this simple
    // implementation.
    // To improve that, pass **head instead
    // of *head
    return -1;

  llnode_t *prev = NULL, *next = NULL;

  find ( head, value, &prev, &next );

  llnode_t *new = (llnode_t*)malloc( sizeof(llnode_t) );
  if ( new == NULL )
    // signals a problem in mem alloc
    return -2;
  
  new->data = value;
  new->prev = prev;
  new->next = next;
  if( prev != NULL )
    prev->next = new;
  if( next != NULL )
	next->prev = new;

  return 0;
}



#if defined(_OPENMP)


// ······················································


int find_and_insert_parallel( llnode_t *head, int value, int use_taskyield, unsigned int *clashes, llnode_t **new_node )
{
  
  if ( head == NULL )
    return 0;

  llnode_t *start = head;
  int done = 0;  

  while ( !done )
    {

      llnode_t *prev = NULL, *next = NULL;
      // find the first guess for the insertion point
      // NOTE: this look-up runs while other threads may be
      //       inserting nodes; as such, if you run with
      //       a thread-analyzer, it may result in
      //       "possible data race" warnings
      find ( start, value, &prev, &next );
  
      // to our best knowledge, next is the first node with data > value
      // and prev is the last node with data < value
      // then, we should create a new node between prev and ptr
  
      // acquire the lock of prev and next
      //

      int locks_acquired = 0;
      while( !locks_acquired )
	{
	  if( prev != NULL )
	    {
	      // acquire the lock on the prev first
	      while ( omp_test_lock(&(prev->lock)) == 0 ) {	    
		if ( use_taskyield ) {
		 #pragma omp taskyield
		} } 

	      SET_OWNER(prev, me);
	      locks_acquired = 1;
	    }

	  // at this point the lock on prev is acquired
      
	  if ( next != NULL )
	    {
	      // we acquire the lock on the next
	      //
	      locks_acquired = my_test_lock(next, me);
	      if( !locks_acquired && (prev!=NULL) )
		{
		  // we did not acquire it, but we have
		  // the lock on prev;
		  // let's release it and retry
		  my_unset_lock(prev);
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
  

     #define MAX_ATTEMPTS 100
      int too_many_attempts = 0;
      // meanwhile, did somebody already insert a node between prev and next?
      if( ( (prev != NULL) && (prev-> next != next) ) ||
	  ( (next != NULL) && (next-> prev != prev) ) )
	{
	  // yes, that happened
	  // let's keep track of how many clashes
	  // 
	 #pragma omp atomic update
	  (*clashes)++;
      
	  if( prev != NULL )  // then, (prev-> next != next)
	    {
	      // the next pointer has changed
	      // prev is not null, so that is our still valid point
	      // we'll walk ahead from there
	      //

	      if (next != NULL) {
		// free the lock on the old next
		my_unset_lock(next); }
	  
	      // search again for a right node, while always keeping prev locked
	      next = prev->next;
	      while( (next != 0x0 ) && (!too_many_attempts) )
		{

		  int attempts      = 0;
		  while (++attempts < MAX_ATTEMPTS)
		    {
		      int got_next_lock = my_test_lock(next, me);
		      if( (!got_next_lock) && (use_taskyield) ) {
		       #pragma omp taskyield
		      }
		    }

		  too_many_attempts = (attempts == MAX_ATTEMPTS);
		  
		  if ( !too_many_attempts )
		    {
		      if( next->data >= value )
			// we have found our new next
			// exit this while
			break;
		      
		      // the current next node is actually
		      // a left node; release the lock on prev
		      my_unset_lock(prev);
		      
		      // and continue searching for a right node
		      // NOTE: we keep the lock on the current "next",
		      // that in the following line becomes the "prev"
		      prev = next;
		      // walk ahead
		      next = next->next;
		    }
		}
	    }
      
	  else if ( next!= NULL ) // and then also (next->prev != prev)
	    // note that next can not be NULL
	    {
	      // the prev pointer has changed
	      // next is not null, so that is our still valid point
	      // we walk back from there
	      //
	      if (prev != NULL) {
		// free the lock on the old prev
		my_unset_lock(prev); }

	      // search again, while always keeping prev locked
	      prev = next->prev;
	      while( (prev != 0x0) && (!too_many_attempts) ) 
		{
		  
		  int attempts = 0;
		  while (++attempts < MAX_ATTEMPTS)
		    {
		      int got_prev_lock = my_test_lock(prev, me);
		      if( (!got_prev_lock) && (use_taskyield) ) {
		       #pragma omp taskyield
		      }
		    }

		  too_many_attempts = (attempts == MAX_ATTEMPTS);
		  
		  if ( !too_many_attempts )
		    {
		      if( prev->data <= value )
			// we have found our new prev
			// exit the while
			break;
		      
		      // ops, the current prev is actually
		      // a left node; release the lock on next
		      my_unset_lock(next);
		      
		      // continue searching for a left node.
		      // NOTE: we keep the lock on the current "prev",
		      // that in the following line becomes the "next"
		      next = prev;
		      // walk backward
		      prev = prev->prev;
		    }
		  
		}
	    }
	  else
	    {
	      printf("Some serious error occurred: "
		     "a prev = next = NULL situation "
		     "has been found!\n");	  
	      return -3;
	    }

	}
      
      if ( too_many_attempts )
	{
	  // we'll release everything and re-start from
	  // finding our insertion point
	  // let's set the start pointer for the find()
	  // routine to a close point
	  //
	  if ( prev!=NULL )
	    { start = prev; my_unset_lock(prev); }
	  if ( next!=NULL )
	    { start = next; my_unset_lock(next); }
	  
	  // if something is left to be done,
	  // just do it
	  if (use_taskyield ) {
	   #pragma omp taskyield
	  }
	}
      
      else
	{
      
	  // at this point, we have found our new left and right pointers,
	  // and we have their locks  
	  //
	  // insertion code
	  //

	  // allocate a new node
	  // HINT: transform everything in an array of nodes
	  //
	  llnode_t *new = (llnode_t*)malloc( sizeof(llnode_t) );
	  *new_node = new;
	  if ( new == NULL )
	    return -2;
  
	  // initialize the new node
	  //omp_init_lock_with_hint( &(new->lock), omp_lock_hint_contended );
	  // if the compiler does not support lock_with_hint,
	  // use the unhinted initialization
	  omp_init_lock( &(new->lock) );
	  SET_OWNER(new, me);
	  new->data = value;
	  new->prev = prev;
	  new->next = next;
	  if ( prev != NULL )
	    prev->next = new;
	  if ( next != NULL)
	    next->prev = new;
  
	  // release locks
	  //
	  if ( prev != NULL )
	    my_unset_lock(prev);
					  
	  if( next != NULL )
	    my_unset_lock(next);
	  
	  done = 1;
	}
      #undef MAX_ATTEMPTS
    }  // closes main while ( !done )
  
  return 0;
}

#endif


// ······················································

int main ( int argc, char **argv )
{
  int N, mode;
  
  {
    int a = 1;
    N    = ( argc > 1 ? atoi(*(argv+a++)) : 1000000 ); 
   #if defined(_OPENMP)
    mode = ( argc > a ? atoi(*(argv+a++)) : DONT_USE_TASKYIELD );
   #endif
    long int seed = ( argc > a ? atoi(*(argv+a++)) : 98765 );

    if ( seed == 0 )
      seed = time(NULL);
    srand48( seed );
  }


  llnode_t *head = (llnode_t*)malloc(sizeof(llnode_t));
  head->data = lrand48();
  head->prev = NULL;
  head->next = NULL;
 #if defined(_OPENMP)
  //omp_init_lock_with_hint( &(head->lock), omp_lock_hint_contended );
  // if the compiler does not support lock_with_hint,
  // use the unhinted initialization
  omp_init_lock( &(head->lock) );
 #endif
  
  ull timing   = CPU_TIME;
  int progress = (N/10);
  int norm     = N*N;
  
 #if !defined(_OPENMP)
  int n = 1;
  while ( n < N )
    {
      int new_value = lrand48() % norm;
      int ret = find_and_insert( head, new_value );
      if ( ret < 0 )
	{
	  printf("I've got a problem inserting node %d\n", n);
	  // cleanup
	  delete_all( head );
	}
      n++;
      if ( n % progress == 0 )
	printf("%.1f%% of nodes done\n", (double)n/N*100);
    }

 #else

  unsigned int clashes = 0;
 #pragma omp parallel
  {
    me = omp_get_thread_num();

    #pragma omp single
    {
      printf("running with %d threads, thread %d generates tasks\n",
	     omp_get_num_threads(), me);

      llnode_t *start = head;
      
      int nsteps   = 10;
      int step     = (N/10);
      int nextstep = 1;

      int N_1      = N-1;
      int n        = 0;

      while ( n < N_1 )
	{
	 #define BATCH_SIZE 100
	  int this_batch_size = ( BATCH_SIZE > N-n-1 ? N-n-1 : BATCH_SIZE );

	 #pragma omp task
	  {
	    for ( int batch = 0; batch < this_batch_size; batch++ )
	      {
		int new_value = lrand48() % norm;
		find_and_insert_parallel( start, new_value, mode, &clashes, &start );
	      }
	  }
	  n += this_batch_size;
	  if ( n >= (step*nextstep)  ) {
	    printf("%.1f%% of nodes done\n", (double)n/N*100);
	    nextstep++; }
	}
    }

    /*
   #pragma omp single
    {
      printf("running with %d threads, thread %d generates tasks\n",
	     omp_get_num_threads(), me);

      llnode_t *node_start = head;
      int n = 1;

      while ( n < N )
	{	  
	  int new_value = lrand48() % norm;

	 #pragma omp task
	  find_and_insert_parallel( node_start, new_value, mode, &clashes, &node_start );
	  
	  n++;
	  if ( n % progress == 0 )
	    printf("%.1f%% of nodes (%d of %d) done\n",
		   (double)n/N*100, n, N);
	}
    }
    */
  }

 #endif

  timing = CPU_TIME - timing;

  printf("verifying..\n"); fflush(stdout);
  
  head = get_head( head );

  int actual_nodes = walk( head, DEBUG_SET );
  if ( actual_nodes != N )
    printf("shame on me! %d nodes instaed of %d have been found!",
	   actual_nodes, N);
  else
    printf("ok\n");

  // cleanup
  delete_all ( head );

  char string[23] = {0};
 #if defined(_OPENMP)
  sprintf( string, " with %u clashes", clashes);  
 #endif
  printf("generation took %g seconds (wtime) %s\n", ((double)timing/1e9), string);
  
  
  return 0;
}
