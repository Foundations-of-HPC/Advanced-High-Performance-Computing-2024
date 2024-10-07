
#if !defined(_TIME_H)
// if time.h was not yet included, do it
//
#include <time.h>
#endif


/* ·········································································
 *
 *  CPU TIME for process
 */

// return process cpu time
#define PCPU_TIME ({struct timespec ts;					\
      (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
       (double)ts.tv_nsec * 1e-9);})

// return process cpu time with a long double
#define PCPU_TIME_L ({struct timespec ts;				\
      (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (long double)ts.tv_sec + \
       (long double)ts.tv_nsec * 1e-9);})


/* ·········································································
 *
 *  CPU TIME for thread
 */

// return thread cpu time
#define TCPU_TIME ({struct timespec ts;				\
      (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
       (double)ts.tv_nsec * 1e-9);})

// return thread cpu time with a long double
#define TCPU_TIME_L ({struct timespec ts;				\
      (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &ts ), (long double)ts.tv_sec + \
       (long double)ts.tv_nsec * 1e-9);})


/* ·········································································
 *
 *  return the number of nanosecond between two different point,
 *  processing two timespec structures
 *  returns a single unsigned long long
 */

// both TSTART and TSTOP are struct timespec
// for instance returned by clock_gettime
//

#define GET_DELTAT( TSTART, TSTOP ) ({ unsigned long long sec = (((TSTOP).tv_sec) - ((TSTART).tv_sec))*1000000000; \
      unsigned long long nsec = 1000000000-((TSTART).tv_nsec)+((TSTOP).tv_nsec); sec+nsec;})
