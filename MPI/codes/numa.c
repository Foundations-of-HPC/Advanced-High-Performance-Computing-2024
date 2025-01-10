#include "allvars.h"
#include <unistd.h>
#include <limits.h>
#include <sched.h>

map_t     Me = {0};           
MPI_Comm  COMM[HLEVELS];

char *LEVEL_NAMES[HLEVELS] = {"NUMA", "ISLAND", "myHOST", "HOSTS", "WORLD"};

MPI_Aint  win_host_master_size = 0;

MPI_Aint    win_ctrl_hostmaster_size; 
MPI_Win     win_ctrl_hostmaster;      
int         win_ctrl_hostmaster_disp; 
void       *win_ctrl_hostmaster_ptr;

MPI_Aint    win_hostmaster_size;
MPI_Win     win_hostmaster;
int         win_hostmaster_disp;
void       *win_hostmaster_ptr; 

win_t *win_ctrl;

int numa_build_mapping( int, int, MPI_Comm *, map_t *);
int numa_map_hostnames( MPI_Comm *, int, int, map_t *);
int get_cpu_id( void );
int get_socket_id( int );
int compare_string_int_int( const void *, const void * );


int numa_init( int Rank, int Size, MPI_Comm *MYWORLD, map_t *Me )
{

  /* 
   * build up the numa hierarchy
   */
  numa_build_mapping( Rank, Size, MYWORLD, Me );
 
  /*
   * initialize the persistent shared windows
   */ 

  int SHMEMl = Me->SHMEMl;
  MPI_Info winfo;
  MPI_Info_create(&winfo);
  MPI_Info_set(winfo, "alloc_shared_noncontig", "true");

  // -----------------------------------
  // initialize the flow control windows
  // -----------------------------------
  memset( &(Me->win_ctrl), 0, sizeof(win_t) );
  Me->win_ctrl.size = (CTRL_END + Me->Ntasks[SHMEMl])*sizeof(int);
  MPI_Win_allocate_shared(Me->win_ctrl.size, sizeof(int), winfo, *Me->COMM[SHMEMl],
  			  &(Me->win_ctrl.ptr), &(Me->win_ctrl.win));

  
  MPI_Aint wsize = (Me->Rank[myHOST] == 0 ? 2 : 0) * sizeof(int);
  MPI_Win_allocate_shared(wsize, sizeof(int), winfo, *Me->COMM[SHMEMl],
			  &win_ctrl_hostmaster_ptr, &win_ctrl_hostmaster);
  
  Me->scwins = (win_t*)aligned_alloc(128, Me->Ntasks[SHMEMl]*sizeof(win_t) );

  win_ctrl = (win_t*)aligned_alloc(128, Me->Ntasks[SHMEMl]*sizeof(win_t) );

  //get the addresses of all the windows from my siblings
  // at my shared-memory level
  
  for( int t = 0; t < Me->Ntasks[SHMEMl]; t++ )
    {
      MPI_Win_shared_query( Me->win_ctrl.win, t, &(Me->scwins[t].size),
			    &(Me->scwins[t].disp), &(Me->scwins[t].ptr) );
      win_ctrl[t] = Me->scwins[t];
    }

  check_ctrl_win( win_ctrl, Me->scwins, Me->Ntasks[Me->SHMEMl] );
  
  if( Me->Rank[SHMEMl] != 0 )
    MPI_Win_shared_query( win_ctrl_hostmaster, 0, &(win_ctrl_hostmaster_size),
			  &win_ctrl_hostmaster_disp, &win_ctrl_hostmaster_ptr );


  return 0;
}


int check_ctrl_win( win_t *A, win_t *B, int n)
{
  int fails = 0;
  for( int i = 0; i < n; i++ )
    {
      if( A[i].size != B[i].size ) { fails++;
	printf("Task %d has got size diff for w %d : %lld %lld\n", oRank,
	       i, (long long)A[i].size, (long long)B[i].size );}

      if(A[i].ptr != B[i].ptr) { fails++;
	  printf("Task %d has got PTR diff for w %d : %p %p\n", oRank,
	       i, A[i].ptr, B[i].ptr );}
	  
      if(A[i].disp != B[i].disp) { fails++;
	  printf("Task %d has got disp diff for w %d : %lld %lld\n", oRank,
		 i, (long long)A[i].disp, (long long)B[i].disp );}
    }
  return fails;
}


void numa_expose( map_t *Me, int level )  
{
  #define BASIC 0

 #define PER_HOST_INFO         1
 #define PER_HOST_MASTER_and_N PER_HOST_INFO
 #define PER_HOST_DETAILS      (PER_HOST_INFO+2)
 #define PER_HOST_MORE_DETAILS (PER_HOST_INFO+4)

 #define PER_LEVEL_INFO 256
  
  if( Me->Rank[WORLD] == 0 )
    printf("===============================================\n"
	   "  NUMA REPORT\n"
	   "===============================================\n\n"
	   "Levels   : %d\n"
	   "# hosts  : %d\n",
	   Me->MAXl,
	   Me->Nhosts );
  MPI_Barrier( *(Me->COMM[WORLD]) );
  
  if( level >= BASIC )
    {
      if( level & PER_HOST_INFO  )
	/*
	 * per-host infos, level = 1
	 */
	{
	  for( int h = 0; h < Me->Nhosts; h++ )
	    {
	      if( Me->myhost == h )
		{
		  if( (level == PER_HOST_MASTER_and_N) &&
		      (Me->Rank[myHOST] == 0) )
		    {
		      // level = 1
		      //
		      printf(" * Rank %d is master of HOST %d which has %d tasks with SHMEMl %d\n",
			     Me->Rank[WORLD], h, Me->Ntasks[myHOST], Me->SHMEMl );
		    }
		  MPI_Barrier( *(Me->COMM[myHOST]) );
		  
		  if( level >= PER_HOST_DETAILS )
		    {
		      // level >= 3
		      //
		      for( int t = 0; t < Me->Ntasks[myHOST]; t++ )
			{
			  if( Me->Rank[myHOST] == t ) { char buffer[2000];
			    sprintf(buffer, "\tRank %d is task %d in host %d ",
				    Me->Rank[WORLD], Me->Rank[myHOST], h);
			    
			    if( level == PER_HOST_MORE_DETAILS ) {

			      // level == 5
			      sprintf(buffer,"rank %d, host %d, socket %d, cpu %d ",
				     Me->Rank[WORLD], Me->myhost, Me->mysocket, Me->mycpu);

			      sprintf( &buffer[strlen(buffer)], " - Ranks_to_myhost: ");
			      for( int tt = 0; tt < Me->Ntasks[myHOST]; tt++ )
				sprintf(&buffer[strlen(buffer)], "%d ", Me->Ranks_to_myhost[tt]);
			      printf("%s\n", buffer);} }
			  MPI_Barrier( *(Me->COMM[myHOST]) );
			}
		    }
		  fflush(stdout);
		}
	      
	      MPI_Barrier( *(Me->COMM[WORLD]) );
	    }
	}
      if( level & PER_LEVEL_INFO )
	/*
	 * per-level infos
	 */
	{
	  for( int l = Me->MAXl; l >= Me->SHMEMl; l-- )
	    if( Me->Rank[l] == 0 )
	      printf("LEVEL %d (%s) has %d tasks\n",
		     l, LEVEL_NAMES[l], Me->Ntasks[l]);
	  
	}
    }

  if( Me->Rank[WORLD] == 0 ) printf("\n\n");
  MPI_Barrier( *(Me->COMM[WORLD]) );
  
  return;
}

int numa_allocate_shared_windows(  map_t *me, MPI_Aint size, MPI_Aint host_size )
{

  int SHMEMl = me->SHMEMl;
  MPI_Info winfo;

  MPI_Info_create(&winfo);
  MPI_Info_set(winfo, "alloc_shared_noncontig", "true");

  // -----------------------------------
  // initialize the data windows
  // -----------------------------------
  MPI_Aint win_temporary_size;
  MPI_Aint win_final_size;
  
  win_hostmaster_size = host_size;

  win_temporary_size = (Me.Rank[SHMEMl] == 0 ? 2*size : size);
  win_final_size     = size;

  me->win.size = win_temporary_size;
  MPI_Win_allocate_shared(me->win.size, 1, winfo, *me->COMM[SHMEMl], &(me->win.ptr), &(me->win.win));

  me->fwin.size = win_final_size;
  MPI_Win_allocate_shared(me->win.size, 1, winfo, *me->COMM[SHMEMl], &(me->fwin.ptr), &(me->fwin.win));

  MPI_Aint wsize = ( me->Rank[SHMEMl] == 0 ? win_hostmaster_size : 0);
  MPI_Win_allocate_shared(wsize, sizeof(double), winfo, *me->COMM[SHMEMl], &win_hostmaster_ptr, &win_hostmaster);

  me->swins = (win_t*)aligned_alloc(128, me->Ntasks[SHMEMl]*sizeof(win_t) );
  me->swins[me->Rank[SHMEMl]] = me->win;
  
  me->sfwins = (win_t*)aligned_alloc(128, me->Ntasks[SHMEMl]*sizeof(win_t) );
  me->sfwins[me->Rank[SHMEMl]] = me->fwin;
  // get the addresses of all the windows from my siblings
  // at my shared-memory level
  //
  for( int t = 0; t < me->Ntasks[SHMEMl]; t++ ) {    
    MPI_Win_shared_query( me->win.win, t, &(me->swins[t].size), &(me->swins[t].disp), &(me->swins[t].ptr) );
    MPI_Win_shared_query( me->fwin.win, t, &(me->sfwins[t].size), &(me->sfwins[t].disp), &(me->sfwins[t].ptr) ); }
  
  if( me->Rank[SHMEMl] != 0 )
    MPI_Win_shared_query( win_hostmaster, 0, &(win_hostmaster_size), &win_hostmaster_disp, &win_hostmaster_ptr );

  return 0;
}

int numa_shutdown( int Rank, int Size, MPI_Comm *MYWORLD, map_t *me )
{
  // free every shared memory and window
  //
  MPI_Win_free(&(me->win_ctrl.win));
  MPI_Win_free(&(me->win.win));
  MPI_Win_free(&(me->fwin.win));
  
  free(me->sfwins);
  free(me->swins);
  free(me->scwins);
  
  free(me->Ranks_to_myhost);
  
  // free all the structures if needed
  //
  free(me->Ranks_to_host);
  

  // anything else
  //
  // ...

  return 0;
  
}

int numa_build_mapping( int Rank, int Size, MPI_Comm *MYWORLD, map_t *me )
{
  COMM[WORLD] = *MYWORLD;
  
  me->Ntasks[WORLD] = Size;
  me->Rank[WORLD]   = Rank;
  me->COMM[WORLD]   = &COMM[WORLD];

  me->mycpu    = get_cpu_id( );
  me->mysocket = get_socket_id ( me->mycpu );

  // --- find how many hosts we are running on;
  //     that is needed to build the communicator
  //     among the masters of each host
  //
  numa_map_hostnames( &COMM[WORLD], Rank, Size, me );


  me->MAXl = ( me->Nhosts > 1 ? HOSTS : myHOST );

  // --- create the communicator for each host
  //
  MPI_Comm_split( COMM[WORLD], me->myhost, me->mysocket, &COMM[myHOST]);
  MPI_Comm_size( COMM[myHOST], &Size );
  MPI_Comm_rank( COMM[myHOST], &Rank );
  
  me->COMM[myHOST] = &COMM[myHOST];
  me->Rank[myHOST]   = Rank;
  me->Ntasks[myHOST] = Size;

  // with the following gathering we build-up the mapping Ranks_to_hosts, so that
  // we know which host each mpi rank (meaning the original rank) belongs to
  //
  
  MPI_Allgather( &me->myhost, sizeof(me->myhost), MPI_BYTE,
		 me->Ranks_to_host, sizeof(me->myhost), MPI_BYTE, COMM[WORLD] );

  me -> Ranks_to_myhost = (int*)aligned_alloc(4, me->Ntasks[myHOST]*sizeof(int));
  MPI_Allgather( &(me->Rank[WORLD]), sizeof(Rank), MPI_BYTE,
		 me->Ranks_to_myhost, sizeof(Rank), MPI_BYTE, *me->COMM[myHOST]);
  


  // --- create the communicator for the
  //     masters of each host
  //
  //COMM[HOSTS]        = MPI_COMM_NULL;
  me->COMM[HOSTS]    = &COMM[HOSTS];
  me->Ntasks[HOSTS]  = 0;
  me->Rank[HOSTS]    = -1;
  
  if ( me->Nhosts > 1 )
    {
      int Im_host_master = ( (me->Rank[myHOST] == 0) ? 1 : MPI_UNDEFINED);
      MPI_Comm_split( COMM[WORLD], Im_host_master, me->Rank[WORLD], &COMM[HOSTS]);
      //
      // NOTE: by default, the Rank 0 in WORLD is also Rank 0 in HOSTS
      //
      if ( Im_host_master != MPI_UNDEFINED )
	{
	  me->COMM[HOSTS] = &COMM[HOSTS];
	  me->Ntasks[HOSTS] = me->Nhosts;
	  MPI_Comm_rank( COMM[HOSTS], &(me->Rank[HOSTS]));
	}
    }

  
  // --- create the communicator for the
  //     numa node
  //
  MPI_Comm_split_type( COMM[myHOST], MPI_COMM_TYPE_SHARED, me->Rank[myHOST], MPI_INFO_NULL, &COMM[NUMA]);
  me->COMM[NUMA] = &COMM[NUMA];
  MPI_Comm_size( COMM[NUMA], &(me->Ntasks[NUMA]));
  MPI_Comm_rank( COMM[NUMA], &(me->Rank[NUMA]));
  
  // check whether NUMA == myHOST and determine
  // the maximum level of shared memory in the
  // topology
  //
  if ( me->Ntasks[NUMA] == me->Ntasks[myHOST] )
    {
      // collapse levels from NUMA to myHOST
      //
      me->Ntasks[ISLAND] = me->Ntasks[NUMA];  // equating to NUMA as we know the rank better via MPI_SHARED
      me->Rank[ISLAND]   = me->Rank[NUMA];
      me->COMM[ISLAND]   = me->COMM[NUMA];
      
      me->Rank[myHOST]   = me->Rank[NUMA];
      me->COMM[myHOST]   = me->COMM[NUMA];
      me->SHMEMl         = myHOST;
    }
  else
    {
      // actually we do not care for this case
      // at this moment
      printf(">>> It seems that rank %d belongs to a node for which "
	     "    the node topology does not coincide \n", Rank );
      me->SHMEMl = NUMA;
    }

  int check_SHMEM_level = 1;
  int globalcheck_SHMEM_level;
  int globalmax_SHMEM_level;
  MPI_Allreduce( &(me->SHMEMl), &globalmax_SHMEM_level, 1, MPI_INT, MPI_MAX, *MYWORLD );

  check_SHMEM_level = ( (me->SHMEMl == myHOST) && (globalmax_SHMEM_level == me->SHMEMl) );
  
  MPI_Allreduce( &check_SHMEM_level, &globalcheck_SHMEM_level, 1, MPI_INT, MPI_MAX, *MYWORLD );
  
  if( globalcheck_SHMEM_level < 1 )
    {
      if( Rank == 0 ) {
	printf("There was an error in determining the topology hierarchy, "
	       "SHMEM level is different for different MPI tasks\n");
	return -1; }
    }  
  
  return 0;  
}


int numa_map_hostnames( MPI_Comm *MY_WORLD,   // the communicator to refer to
			int Rank,              // the initial rank of the calling process in MYWORLD
			int Ntasks,            // the number of tasks in MY_WORLD
			map_t *me)             // address of the info structure for the calling task

{
  // --------------------------------------------------
  // --- init some global vars
  me -> Ranks_to_host = (int*)aligned_alloc(4, Ntasks*sizeof(int));
  me -> Nhosts = 0;
  me -> myhost = -1;

  // --------------------------------------------------
  // --- find how many hosts we are using
  

  char myhostname[HOST_NAME_MAX+1];
  gethostname( myhostname, HOST_NAME_MAX+1 );


  // determine how much space to book for hostnames
  int myhostlen = strlen(myhostname)*2;            // *2 is just to keep the alignment
						   // for the rank integer in the structure hostname_rank_t


 #if !defined(__clang__)
  int maxhostlen = 0;
  MPI_Allreduce ( &myhostlen, &maxhostlen, 1, MPI_INT, MPI_MAX, *MY_WORLD );
 #else
  {
   #define maxhostlen 500

    int get_maxhostlen = 0;
    MPI_Allreduce ( &myhostlen, &get_maxhostlen, 1, MPI_INT, MPI_MAX, *MY_WORLD );
    if( get_maxhostlen > 500 )
      printf("the possible lenght of hostname %d is larger than the hardcoded value %d;"
	     " recompile using -DMAXHOSTLEN=%d\n",
	     get_maxhostlen, maxhostlen, get_maxhostlen);
  }
 #endif

  // collect hostnames
  //
  typedef struct {
    int rank;
    char hostname[maxhostlen];    
  } hostname_rank_t;
      
  hostname_rank_t mydata;
  //hostname_rank_t *alldata = (hostname_rank_t*)aligned_alloc(16, Ntasks*sizeof(hostname_rank_t) );
  hostname_rank_t *alldata = (hostname_rank_t*)malloc(Ntasks*sizeof(hostname_rank_t) );
  memset( alldata, 0, Ntasks*sizeof(hostname_rank_t));

  mydata.rank = Rank;  
  snprintf( mydata.hostname, maxhostlen, "%s", myhostname);

  MPI_Allgather( &mydata, sizeof(hostname_rank_t), MPI_BYTE, alldata, sizeof(hostname_rank_t), MPI_BYTE, *MY_WORLD );
  
  // sort the hostnames
  //       1) set the lenght of string for comparison
  int dummy = maxhostlen;
  compare_string_int_int( NULL, &dummy );


  //       2) actually sort
  qsort( alldata, Ntasks, sizeof(hostname_rank_t), compare_string_int_int );
  // now the array alldata is sorted by hostname, and inside each hostname the processes
  // running on each host are sorted by their node, and for each node they are sorted
  // by ht.
  // As a direct consequence, the running index on the alldata array can be considered
  // as the new global rank of each process
  
  // --- count how many diverse hosts we have, and register each rank to its host, so that
  //      we can alway find all the tasks with their original rank

  char *prev = alldata[0].hostname;
  for ( int R = 0; R < Ntasks; R++ )
  {	
    if ( strcmp(alldata[R].hostname, prev) != 0 ) {      
      me->Nhosts++; prev = alldata[R].hostname; }

    if ( alldata[R].rank == Rank )        // it's me
      me->myhost = me->Nhosts;            // remember my host
  }
  me->Nhosts++;

  free( alldata );

  return me->Nhosts;
}



int compare_string_int_int( const void *A, const void *B )
// used to sort structures made as
// { char *s;
//   int b;
//   ... }
// The sorting is hierarchical by *s first, then b
//   if necessary
// The length of *s is set by calling
//   compare_string_int_int( NULL, len )
// before to use this routine in qsort-like calls
{
  static int str_len = 0;
  if ( A == NULL )
    {
      str_len = *(int*)B;
      return 0;
    }

  // we do not use strncmp because str_len=0,
  // i.e. using this function without initializing it,
  // can be used to have a sorting only on
  // strings
  char *As = (char*)((int*)A+1);
  char *Bs = (char*)((int*)B+1);
  int order = strcmp( As, Bs );
  
  if ( str_len && (!order) )
    {
      int a = *(int*)A;
      int b = *(int*)B;
      order = a - b;
      /* if( !order ) */
      /* 	{ */
      /* 	  int a = *((int*)((char*)A + str_len)+1); */
      /* 	  int b = *((int*)((char*)B + str_len)+1); */
      /* 	  order = a - b; */
      /* 	} */
    }
  
  return order;
}


#define CPU_ID_ENTRY_IN_PROCSTAT 39

int read_proc__self_stat( int, int * );

int get_cpu_id( void )
{
  
#if defined(_GNU_SOURCE)                              // GNU SOURCE ------------
  
  return  sched_getcpu( );

#else

#ifdef SYS_getcpu                                     //     direct sys call ---
  
  int cpuid;
  if ( syscall( SYS_getcpu, &cpuid, NULL, NULL ) == -1 )
    return -1;
  else
    return cpuid;
  
#else

  int val;
  if ( read_proc__self_stat( CPU_ID_ENTRY_IN_PROCSTAT, &val ) == -1 )
    return -1;

  return (int)val;

#endif                                                // -----------------------
#endif

}


int get_socket_id( int cpuid )
{
  
  FILE *file = fopen( "/proc/cpuinfo", "r" );
  if (file == NULL )
    return -1;

  char *buffer = NULL;
  int socket = -1;
  int get_socket = 0;
  
  while ( socket < 0 )
    {
      size_t  len;
      ssize_t read;
      read = getline( &buffer, &len, file );
      if( read > 0 ) {
	switch( get_socket ) {
	case 0: {	if( strstr( buffer, "processor" ) != NULL ) {
	      char *separator = strstr( buffer, ":" );
	      int   proc_num  = atoi( separator + 1 );
	      get_socket = (proc_num == cpuid ); } } break;
	case 1: { if( strstr( buffer, "physical id" ) != NULL ) {
	      char *separator = strstr( buffer, ":" );
	      socket    = atoi( separator + 1 ); } } break;
	}} else if ( read == -1 ) break;
    }    
  
  fclose(file);
  free(buffer);

  return socket;
}


int read_proc__self_stat( int field, int *ret_val )
/*
  Other interesting fields:

  pid      : 0
  father   : 1
  utime    : 13
  cutime   : 14
  nthreads : 18
  rss      : 22
  cpuid    : 39

  read man /proc page for fully detailed infos
 */
{
  // not used, just mnemonic
  // char *table[ 52 ] = { [0]="pid", [1]="father", [13]="utime", [14]="cutime", [18]="nthreads", [22]="rss", [38]="cpuid"};

  *ret_val = 0;

  FILE *file = fopen( "/proc/self/stat", "r" );
  if (file == NULL )
    return -1;

  char   *line = NULL;
  int     ret;
  size_t  len;
  ret = getline( &line, &len, file );
  fclose(file);

  if( ret == -1 )
    return -1;

  char *savetoken = line;
  char *token = strtok_r( line, " ", &savetoken);
  --field;
  do { token = strtok_r( NULL, " ", &savetoken); field--; } while( field );

  *ret_val = atoi(token);

  free(line);

  return 0;
}
