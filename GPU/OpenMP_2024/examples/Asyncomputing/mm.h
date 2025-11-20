#define TCPU_TIME (clock_gettime( id, &ts ), (double)ts.tv_sec + \
																			(double)ts.tv_nsec * 1e-9)

#define TYPE double
#if !defined (MCONST) && !defined (MPROG)
#define MCONST
#endif

/*
* CPU gpu defintions
*/
#define STRINGIFY(a) #a
#define ACC_APPLY_PRAGMA(...)  _Pragma(STRINGIFY(__VA_ARGS__))
#if defined (GPU)
	#define ACC_PRAGMA(...) ACC_APPLY_PRAGMA( omp __VA_ARGS__)
	#define ACC_FUNCTION_BEGIN(...) _Pragma("omp declare target")
	#define ACC_FUNCTION_END(...) _Pragma("omp end declare target")
#elif defined (CPU)
	#define ACC_PRAGMA(...) ACC_APPLY_PRAGMA( omp __VA_ARGS__)
	#define ACC_FUNCTION_BEGIN(...) _Pragma("omp declare target")
	#define ACC_FUNCTION_END(...) _Pragma("omp end declare target")
#else
	#define ACC_PRAGMA(...)
	#define ACC_FUNCTION_BEGIN(...)
	#define ACC_FUNCTION_END(...)
#endif



double errsqr(int Ndim, int Mdim, TYPE *C, TYPE *Cref);
void mm_zero(TYPE *C, int Ndim, int Mdim );
void mm_rand(TYPE *C, int Ndim, int Mdim );
void mm_print(TYPE *C, int Ndim, int Mdim );
void mm_init(TYPE *A, int Ndim, int Mdim, TYPE val);

void __attribute__ ((noinline)) mm_mul_gpu(TYPE *A, TYPE *B, TYPE *C, TYPE alpha, int Ndim, int Mdim, int Kdim);
void __attribute__ ((noinline)) mm_mul_gpu2(TYPE *A, TYPE *B, TYPE *C, TYPE alpha, int Ndim, int Mdim, int Kdim);


void __attribute__ ((noinline)) mm_mul_cpu(TYPE *A, TYPE *B, TYPE *C, TYPE alpha, int Ndim, int Mdim, int Kdim);

