/* ============================================================================
 * EXAMPLE 3: Complex Control Flow
 * ============================================================================
 * 
 * PURPOSE: Show how complex conditionals and early exits prevent vectorization.
 *
 * LEARNING GOALS:
 * - Understand "not vectorized: control flow in loop"
 * - Distinguish between vectorizable and non-vectorizable conditionals
 * - Learn when to restructure code vs. accept scalar execution
 *
 * COMPILE WITH:
 *   gcc -O3 -march=native -ftree-vectorize -fopt-info-vec-all 03_control_flow.c -o control
 *
 * EXPECTED OUTPUT (GCC):
 *   03_control_flow.c:43:5: missed: couldn't vectorize loop
 *   03_control_flow.c:43:5: missed: not vectorized: control flow in loop.
 *   03_control_flow.c:45:9: missed: statement clobbers memory: <bb 6>:
 *
 * KEY OBSERVATIONS:
 * - "control flow in loop" = conditionals that can't be masked
 * - Early exit (break) is particularly problematic
 * - Function calls inside conditionals make it worse
 *
 * ============================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#if defined(__clang__)
   #define ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
   #define ASSUME(cond) do { if (!(cond)) __builtin_unreachable(); } while (0)
#else
   #define ASSUME(cond) ((void)0)
#endif

#define N 1000000


#if !defined(BETTER_FIX)
#warning "no better fix"

/* ============================================================================
 * FIXABLE VERSION: Remove early exit, fully vectorizable
 *          NOTE!!  Semantically equivalent to the non-fix version
 * ============================================================================
 */
int find_threshold_breach(const double *data, int n, double threshold)
{
    int result = n;    
    double threshold2 = 2*threshold;
    
    // No early exit - all iterations execute
    // Simple conditional - can use blend/select instructions
    for (int i = 0; i < n; i++)
      {
	int hint = ( data[i] > threshold2 ? i : n );
	result   = ( result < hint ? hint : result  );
      }

    return (result < n ? result : -1);
}

#elif (BETTER_FIX == 1)

// [ QUESTION ]
// in the following we are using "4", meaning that our target
// vector lenght is 4
// can we generalize ?


int find_threshold_breach(const double *data, int n, double threshold)
{
    int result = n;    
    double threshold2 = 2*threshold;

    int n_vec = (n/4)*4;
    // for GCC it seems better
    //int n_vec = n & 0xFFFFFFFC;
    //int n_vec = n & ~((int)3);
    
    // now it is clear that n_vec is
    // a multiple of 4
    for (int i = 0; i < n_vec; i++)
      {
	int hint = ( data[i] > threshold2 ? i : n );
	result   = ( result < hint ? hint : result  );
      }

    for ( int i = n_vec; i < n; i++ )
      {
	int hint = ( data[i] > threshold2 ? i : n );
	result   = ( result < hint ? hint : result  );
      }
	
    
    return (result < n ? result : -1);
}


#elif (BETTER_FIX == 2)

int find_threshold_breach(const double *data, int n, double threshold)
{
    int result = n;    
    double threshold2 = 2*threshold;

    assert( n%4 == 0);
    ASSUME( n%4 == 0)
    
    // now it is clear that n_vec is
    // a multiple of 4
    for (int i = 0; i < n; i++)
      {
	int hint = ( data[i] > threshold2 ? i : n );
	result   = ( result < hint ? hint : result  );
      }
    
    return (result < n ? result : -1);
}



#endif

/* ============================================================================
 * COMPILE THE FIXED VERSION:
 *   gcc -O3 -march=native -ftree-vectorize -fopt-info-vec-all 03_control_flow.c \
 *       -DUSE_FIX -o control_fixed
 *
 * EXPECTED OUTPUT for fixed version:
 *   03_control_flow.c:61:5: optimized: loop vectorized using 32 byte vectors
 *
 * EXPLANATION:
 * - Removed nested if
 * - Removed early exit (break/return)
 * - Simple condition can be converted to mask operation
 * - Compiler can now vectorize with blend instructions
 * ============================================================================
 */

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : N;
    double threshold = 500.0;
    
    double *data = (double*)malloc(n * sizeof(double));
    
    // Initialize with values 0 to n-1
    for (int i = 0; i < n; i++) {
        data[i] = (double)i;
    }
    
    int result = find_threshold_breach(data, n, threshold);
    #if defined(USE_FIX)	
    printf("Fixed version result: %d\n", result);
    #else
    printf("Original version result: %d\n", result);
    #endif
    
    free(data);
    return 0;
}

/* ============================================================================
 * TEACHER NOTES:
 * 
 * Types of Control Flow:
 * 
 * 1. VECTORIZABLE (via masking):
 *    - Simple if-then-else: y = (x > 0) ? a : b
 *    - No early exits
 *    - No function calls inside conditional
 *    - Uniform control flow (all iterations do same amount of work)
 *
 * 2. NON-VECTORIZABLE:
 *    - Early exits (break, return, goto)
 *    - Nested complex conditionals
 *    - Function calls with side effects
 *    - Variable loop counts based on data
 *
 * 3. STRATEGY:
 *    - For search operations: Consider keeping all iterations, update result
 *    - For complex logic: Extract to separate function, vectorize outer loop
 *    - For data-dependent exits: Use sentinel values or restructure algorithm
 *
 * The "fixed" version isn't semantically identical (finds LAST match, not FIRST),
 * but demonstrates the principle. For exact semantics, you'd need a different approach
 * (maybe SIMD intrinsics with horizontal operations).
 * ============================================================================
 */
