
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
#define TIME_CUT 100000000


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
#define TIMESTAMP (CPU_TIME % TIME_CUT)
#define dbgout(...) printf( __VA_ARGS__ );
#else
#define TIMESTAMP
#define dbgout(...) 
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
  int        owner;
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
int       walk            ( llnode_t *);
int       delete_all      ( llnode_t * );
int       find            ( llnode_t *, int, llnode_t **, llnode_t ** );
int       find_and_insert ( llnode_t *, int );

#if defined(_OPENMP)
int       find_and_insert_parallel ( llnode_t *, int, int, unsigned int * );
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

int walk ( llnode_t *start )
/*
 * walk the list starting from the node start
 * as first, the list is walked back until the list head
 * if mode == 1, the list is then walked ahed printing
 * the first 100 nodes.
 */
{
  int n = 0;
  if ( start != NULL )
    {
      n = 1;
      int prev_value = start->data;
      printf("%9d [-]", start->data );
      start = start->next;
      while( start != NULL)
	{
	  if (++n < 100 )
	    printf( "%9d %s ",
		   start->data,
		   (start->data < prev_value? "[!]":"[ok]") );
	  else if ( n == 100)
	    printf( "..." );
	  prev_value = start->data;
	  start = start->next;
	}
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


int find_and_insert_parallel( llnode_t *head, int value, int use_taskyield, unsigned int *clashes )
{
  
  if ( head == NULL )
    return -1;

  dbgout("[ %llu ] > T %d process value %d\n", TIMESTAMP, me, value );

  int done = 0;

  while ( !done )
    {

      llnode_t *prev = NULL, *next = NULL;
      // find the first guess for the insertion point
      //
      find ( head, value, &prev, &next );
  
      dbgout("[ %llu ] T %d V %d found p: %d and n: %d\n", TIMESTAMP, me, value,
	     prev!=NULL?prev->data:-1, next!=NULL?next->data:-1);
  
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

	      prev->owner = me;
	      locks_acquired = 1;
	    }

	  // at this point the lock on prev is acquired
      
	  if ( next != NULL )
	    {
	      // we acquire the lock on the next
	      //
	      locks_acquired = omp_test_lock(&(next->lock));
	      if( !locks_acquired && (prev!=NULL) )
		{
		  // we did not acquire it, but we have
		  // the lock on prev;
		  // let's release it and retry
		  prev->owner = -1;
		  omp_unset_lock(&(prev->lock));
		  if ( use_taskyield ) {
		    // if we wnat to use the taskyield feature,
		    // that is a good moment ?
		   #pragma omp taskyield
		  }
		}
	      else
		next->owner = me;
	    }      
	}
  

      dbgout("[ %llu ] T %d V %d locked: (p: %d %p p>n: %d) (n: %d %p n<p: %d)\n",
	     TIMESTAMP, me, value,
	     (prev!=NULL?prev->data:-1), (prev!=NULL?prev:0),  ((prev!=NULL)&&(prev->next!=NULL)?(prev->next)->data:-1),
	     (next!=NULL?next->data:-1), (next!=NULL?next:0),  ((next!=NULL)&&(next->prev!=NULL)?(next->prev)->data:-1) );

     #define MAX_ATTEMPTS 100
      int attempts = 0;
      // meanwhile, did somebody already insert a node between prev and next?
      if( ( (prev != NULL) && (prev-> next != next) ) ||
	  ( (next != NULL) && (next-> prev != prev) ) )
	{
	  // yes, that happened
	  // let's keep track of how many clashes
	  // 
	 #pragma omp atomic update
	  (*clashes)++;
      
	  if( (prev != NULL) && (prev-> next != next) )
	    {
	      // the next pointer has changed
	      // prev is not null, so that is our still valid point
	      // we'll walk ahead from there
	      //
	      dbgout("[ %llu ]\t>> T %d V %d next has changed: from %d to %d\n",
		     TIMESTAMP, me, value,
		     (next!=NULL?next->data:-1),(prev->next!=NULL?(prev->next)->data:-1) );

	      if (next != NULL) {
		// free the lock on the old next
		next->owner = -1;
		omp_unset_lock(&(next->lock));
		dbgout("[ %llu ]\t>> T %d V %d releases lock on %d %p\n",
		       TIMESTAMP, me, value, next->data, next); }

	      dbgout("[ %llu ]\t\t>>> T %d V %d restart from %d to walk ahead\n",
		     TIMESTAMP, me, value, prev->data);
	  
	      // search again fgor a right node, while always keeping prev locked
	      next = prev->next;
	      while( (next != 0x0) && (attempts < 100) )
		{
		  dbgout("[ %llu ]\t\t\t>>> T %d V %d stepping into %d %p while walking ahead {own: %d}\n",
			 TIMESTAMP, me, value, next->data, next, next->owner );

		  int got_next_lock = omp_test_lock(&(next->lock));
		  if ( got_next_lock )
		    {
		      next->owner = me;
		      dbgout("[ %llu ]\t>> T %d V %d got lock on %d %p while walking ahead\n",
			     TIMESTAMP, me, value, next->data, next);

		      if( next->data >= value )
			// we have found our new next
			// exit this while
			break;

		      // the current next node is actually
		      // a left node; release the lock on prev
		      prev->owner = -1;
		      omp_unset_lock(&(prev->lock));
		      dbgout("[ %llu ]\t>> T %d V %d releases lock on %d %p while walking ahead\n",
			     TIMESTAMP, me, value, prev->data, prev);

		      // and continue searching for a right node
		      // NOTE: we keep the lock on the current "next",
		      // that in the following line becomes the "prev"
		      prev = next;
		      // walk ahead
		      next = next->next;
		    }
		  attempts++;
		}
	      if ( (attempts == MAX_ATTEMPTS) &&
		   (prev!=NULL) ) {
		dbgout("[ %llu ]\t>> T %d V %d releases lock on %d %p after MAX_ATTEMPTS while walking back\n",
		       TIMESTAMP, me, value, prev->data, prev);
		omp_unset_lock(&(prev->lock)); }	      
	    }
      
	  else if ( next->prev != prev )
	    // note that next can not be NULL
	    {
	      // the prev pointer has changed
	      // next is not null, so that is our still valid point
	      // we walk back from there
	      //
	      dbgout("[ %llu ]\t>> T %d V %d prev has changed: from %d to %d\n",
		     TIMESTAMP, me, value,
		     (prev!=NULL?prev->data:-1),(next->prev!=NULL?(next->prev)->data:-1) );

	      if (prev != NULL) {
		// free the lock on the old prev
		prev->owner = -1;
		dbgout("[ %llu ]\t>> T %d V %d releases lock on %d %p while walking back\n",
		       TIMESTAMP, me, value, prev->data, prev);
		omp_unset_lock(&(prev->lock)); }

	      dbgout("[ %llu ]\t\t>> T %d V %d restart from %d %p to walk back\n",
		     TIMESTAMP, me, value, next->data, next);
      	
	      // search again, while always keeping prev locked
	      prev = next->prev;
	      while( (prev != 0x0) && (attempts < 100) )
		{
		  dbgout("[ %llu ]\t\t\t>>> T %d V %d stepping into %d %p while walking back {own: %d}\n",
			 TIMESTAMP, me, value, prev->data, prev, prev->owner);
		  int got_prev_lock = omp_test_lock(&(prev->lock));
		  if ( got_prev_lock )
		    {
		      prev->owner = me;
		      dbgout("[ %llu ]\t>> T %d V %d got lock on %d %p while walking back\n",
			     TIMESTAMP, me, value, prev->data, prev);
		 
		      if( prev->data <= value )
			// we have found our new prev
			// exit the while
			break;

		      // ops, the current prev is actually
		      // a left node; release the lock on next
		      next->owner = -1;
		      omp_unset_lock(&(next->lock));
		      dbgout("[ %llu ]\t>> T %d V %d releases lock on %d %p while walking back\n",
			     TIMESTAMP, me, value, next->data, next);

		      // continue searching for a left node.
		      // NOTE: we keep the lock on the current "prev",
		      // that in the following line becomes the "next"
		      next = prev;
		      // walk backward
		      prev = prev->prev;
		    }
		  attempts++;
		}
	      if ( (attempts == MAX_ATTEMPTS) &&
		   (next!=NULL) ) {
		dbgout("[ %llu ]\t>> T %d V %d releases lock on %d %p after MAX_ATTEMPTS while walking back\n",
		       TIMESTAMP, me, value, next->data, next);
		omp_unset_lock(&(next->lock)); }
		
	    }
	  else if ( next == NULL )
	    {
	      printf("Some serious error occurred, a prev = next = NULL situation arose!\n");	  
	      return -3;
	    }
	}

      if ( attempts < MAX_ATTEMPTS )
	{
      
	  // at this point, we have found our new left and right pointers,
	  // and we have their locks  
	  //
	  // insertion code
	  //

	  dbgout("[ %llu ]\tthread %d processing %d inserts value between %d %p and %d %p\n",
		 TIMESTAMP, me, value,
		 (prev!=NULL?prev->data:-1), (prev!=NULL?prev:0),
		 (next!=NULL?next->data:-1), (next!=NULL?next:0) );
  
	  // allocate a new node
	  // HINT: transform everything in an array of nodes
	  //
	  llnode_t *new = (llnode_t*)malloc( sizeof(llnode_t) );
	  if ( new == NULL )
	    return -2;
  
	  // initialize the new node
	  //omp_init_lock_with_hint( &(new->lock), omp_lock_hint_contended );
	  omp_init_lock( &(new->lock) );
	  omp_set_lock( &(new->lock) );   // well not so usefule, though
	  new->owner= me;
	  new->data = value;
	  new->prev = prev;
	  new->next = next;
	  if ( prev != NULL )
	    prev->next = new;
	  if ( next != NULL)
	    next->prev = new;
  
	  // release locks
	  //
	  if ( prev != NULL ) {
	    prev->owner = -1;
	    omp_unset_lock(&(prev->lock));
	    dbgout("[ %llu ]\tthread %d processing %d releases lock on %d %p\n",
		   TIMESTAMP, me, value, prev->data, prev);}
					  
	  if( next != NULL ) {
	    next->owner = -1;
	    omp_unset_lock(&(next->lock));
	    dbgout("[ %llu ]\tthread %d processing %d releases lock on %d %p\n",
		   TIMESTAMP, me, value, next->data, next);}

	  new->owner = -1;
	  omp_unset_lock(&(new->lock));
	  dbgout("[ %llu ]\tthread %d processing %d releases lock on %d %p\n",
		 TIMESTAMP, me, value, new->data, new);

	  done = 1;
	}
      #undef MAX_ATTEMPTS
    }  // closes main while ( !done )
  
  dbgout("T %d V %d has done\n", me, value);
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
    int seed = ( argc > a ? atoi(*(argv+a++)) : 98765 );
    
    srand48( seed );
  }


  llnode_t *head = (llnode_t*)malloc(sizeof(llnode_t));
  head->data = lrand48();
  head->prev = NULL;
  head->next = NULL;
 #if defined(_OPENMP)
  //omp_init_lock_with_hint( &(head->lock), omp_lock_hint_contended );
  omp_init_lock( &(head->lock) );
 #endif
  
  ull timing = CPU_TIME;
  int progress = (N/10);
  
 #if !defined(_OPENMP)
  int n = 1;
  while ( n < N )
    {
      int new_value = lrand48();
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
      int n = 1;

      while ( n < N )
	{	  
	  int new_value = lrand48() % (N/10);  // we want to have some clashes

	 #pragma omp task
	  find_and_insert_parallel( head, new_value, mode, &clashes );
	  
	  n++;
	  if ( n % progress == 0 )
	    printf("%.1f%% of nodes done\n", (double)n/N*100);
	}
    }
  }

 #endif

  timing = CPU_TIME - timing;

  head = get_head( head );

  int actual_nodes = walk( head);
  if ( actual_nodes != N )
    printf("shame on me! %d nodes instaed of %d have been found!",
	   actual_nodes, N);

  // cleanup
  delete_all ( head );

  char string[23] = {0};
 #if defined(_OPENMP)
  sprintf( string, " with %u clashes", clashes);  
 #endif
  printf("generation took %g seconds (wtime) %s\n", ((double)timing/1e9), string);
  
  
  return 0;
}
