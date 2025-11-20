

#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include <omp.h>

#define CPU_TIME_W ({ struct timespec ts; (clock_gettime( CLOCK_REALTIME, &ts ), \
                                           (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9); })
#define CPU_TIME_T ({ struct timespec myts; (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), \
                                             (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9); })


typedef unsigned int count_t;
typedef unsigned long long int ull;

// the structure which represents "a particle"

typedef struct {
  int time;           // <-- this is the particle's own time, to be updated
  int updated_by;     // just to record who has down the work
  omp_lock_t lock;    
 } part_t;


int Nthreads;
int myid;
unsigned short int seed[3];
#pragma omp threadprivate(myid, seed)


void update_particle( part_t *P, int time )
//
//  this function "updates" a given particle *P to
//  time "time"
{
  // mimic some hard work
  const struct timespec wait={0, 1000+nrand48(seed) % 50000};
  nanosleep(&wait, NULL);

  // record who has done the work
 #pragma omp write release
  P->updated_by = myid;
  
  // finally update the time
 #pragma omp write release
  P->time = time;
  
  return;
}


void use_particle( part_t *P )
{
  // just mimic some work
  //  
  const struct timespec wait={0, 1000+nrand48(seed) % 50000};
  nanosleep(&wait, NULL);
  return;
}


int compare ( const void *A, const void *B )
// used to sort the list of particles processed by every thread
//
{
  count_t a = *(count_t*)A;
  count_t b = *(count_t*)B;

  return ( a > b) - (a < b);
}



int main ( int argc, char **argv )
{
  int max_current_time_of_particles = 100;
  
  count_t N            = ( argc>1 ? (count_t)atoll(*(argv+1)) : 200000 );                           // how many porticles in total
  int     update_time  = ( argc>2 ? atoi(*(argv+2)) : (int)(max_current_time_of_particles*0.8) );   // this is the current time; all the particles
												    // will be initialized having their proper time smaller
												    // than max_current_time_of_particles; hence this value
												    // controls how many particles will be in need of an
												    // update
  int     sort_my_list = ( argc>3 ? atoi(*(argv+3)) : 1 );                                          // every thread will pick up some amount of particles
												    // to process, randomòy chosen; sorting the list of the
												    // particles (default choice) will maximize the # of
												    // clashes

  
  part_t *P = (part_t*)malloc(sizeof(part_t)* N );
  double  wallclock = 0;
  
  #pragma omp parallel
  {

    // ·····················································································
    // initialize
    // ·····················································································    
    //
   #pragma omp single
    Nthreads = omp_get_num_threads();
   #pragma omp barrier                   // just to ensure that everyone gets Nthreads correctly
    
    myid = omp_get_thread_num();
    
    seed[0] = myid;
    seed[1] = myid & 123;
    seed[2] = myid*11;
    
   #pragma omp for schedule(static)
    for ( count_t p = 0; p < N; p++ )
      {
	P[p].time       = nrand48( seed ) % max_current_time_of_particles;
	P[p].updated_by = -1;
	omp_init_lock(&P[p].lock);
      }


   #pragma omp single
    printf("running with %llu particles and %d threads\n", (ull)N, Nthreads );

    
    // 
    // generate a random list of particles that every thread processes;
    // the minimum amount is  0.5* N/Nthreds
    // we add on top a random quantity up to 20% of N
    //
    count_t  myN     = (count_t)((double)N / Nthreads / 2) + nrand48( seed ) % (N/5);
    count_t *my_list = (count_t*)malloc(sizeof(count_t)*myN);

    for ( count_t i = 0; i < myN; i++ )
      {
	count_t idx = 0;
	while ( idx == 0 )
	  {
	    idx = nrand48(seed) % N;
	    // check that idx is not already present
	    int j = 0;
	    while ( (j<i) && (my_list[j++] != idx) );

	    if ( j < i )
	      idx = 0;
	  }

	my_list[i] = idx;	
      }

    if ( sort_my_list )
      // sorting the list of processed particles will maximize the clashes
      qsort ( my_list, myN, sizeof(count_t), compare );

    printf("th %1d has %llu active particles\n", myid, (ull)myN);

    #pragma omp barrier

   #pragma omp single
    wallclock = CPU_TIME_W;

    // ·····················································································
    // processing
    // ·····················································································
    //
    double mytime = CPU_TIME_T;
    
                                   // variables used to collect some stats
    count_t already_updated = 0;   // records how many particles were updated already
    count_t spinning        = 0;   // records how many particles I had to wait for
  
    int     block = 1;             // used to print progress on the output
    
    for ( count_t i = 0; i < myN; i++ )
      {
	// print % progress
	if ( 10*((double)i / myN) > block )
	  printf("thread %2d has done %2d%%\n", myid, block*10), block++;

	count_t idx = my_list[i];
	// get the current time of i-th particle
	int _time_;
       #pragma omp atomic read acquire
	_time_ = P[idx].time;
	
	if ( _time_ < update_time )
	  {
	    // need to update
	    
	    int got_it = omp_test_lock( &P[idx].lock );
	    if ( got_it ) 
	      {
	       #pragma omp atomic read acquire
		_time_ = P[idx].time;		
		if ( _time_ < update_time )
		  // we check again the time value to exclude the case
		  // that we acquired the lock while another thread
		  // was finishing the particle drift. Basically to
		  // exclude that we arrived exactly before the update
		  // of time, which is the last operation of
		  // update_particle()
		  {
		    //printf("[U] th %2d updates p %llu\n", myid, (ull)idx);
		    // the particle's update is on me
		    update_particle( &P[idx], update_time );
		  }
		omp_unset_lock( &P[idx].lock );
	      }
	    
	    else
	      {
		/*
		 *  we need to wait for the particle's update because we
		 *  want to "use it"; in fact, that tipically happens in
		 *  a tree walk while calculating forces.
		 *  As such we can not just go and be back
		 *  Alternatively, every thread could have a stack to store
		 *  particles to-be rechecked and accounted..
		 *
		 */

		already_updated ++;
		spinning++;
		// Someone else is doing the update
		
		//#pragma omp atomic update             // restoring 1 is not needed actually
		// P[p].lock--;

		// spin until the update has ended
		//
		// printf("[S] th %2d spins while waiting p %llu\n", myid, (ull)idx);
		while (_time_ < update_time)
		  {
		    const struct timespec wait={0, 10};
		    nanosleep(&wait, NULL);
		   #pragma omp atomic read acquire
		    _time_ = P[idx].time;
		  }		
	      }
	  }

	use_particle( &P[idx] );
      }

    mytime = CPU_TIME_T - mytime;

   #pragma omp single
    wallclock = CPU_TIME_W - wallclock;

    free( my_list );
    
   #pragma omp barrier
   #pragma omp single
    printf("\n\n");

    // ·····················································································
    // let us know what happened
    // ·····················································································
    //

   #pragma omp for ordered
    for ( int i = 0; i < Nthreads; i++ )
     #pragma omp ordered
      printf("\t thread %2d has run for %g sec\n"
	     "it encountered %llu already-updated particles and waited for %llu being-updated particles\n",
	     myid, mytime, (ull)already_updated, (ull)spinning );
        
  }

  printf("wall-clock run time for processing is %g sec\n", wallclock );
  free ( P );
  
  return 0;
}
