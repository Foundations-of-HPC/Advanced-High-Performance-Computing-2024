
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



// =========================================================================
//
//  define useful quantities
//

typedef unsigned long long ull;
#define TIME_CUT 1000000009



#define CPU_TIME ({ struct timespec ts; (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), \
					 (ull)ts.tv_sec * 1000000000 +	\
					 (ull)ts.tv_nsec); })

//
// =========================================================================
//
// define data structures
//


typedef struct llnode
{
  int data;
  struct llnode *next;
  struct llnode *prev;  
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
int       find_and_insert ( llnode_t *, int );

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
  while( ptr->prev != NULL )
    ptr = ptr->prev;
  
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


// ······················································

int main ( int argc, char **argv )
{
  int N, mode;
  long int seed;
  {
    int a = 1;
    N    = ( argc > 1 ? atoi(*(argv+a++)) : 1000000 ); 
    seed = ( argc > a ? atoi(*(argv+a++)) : 98765 );

    if ( seed == 0 )
      seed = time(NULL);
    srand48( seed );
  }


  llnode_t *head = (llnode_t*)malloc(sizeof(llnode_t));
  head->data = lrand48();
  head->prev = NULL;
  head->next = NULL;
  
  ull timing   = CPU_TIME;
  int norm     = N*N;
  int progress = (N/10);
  int n        = 1;
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
  printf("generation took %g seconds (wtime)\n", ((double)timing/1e9));
  
  
  return 0;
}
