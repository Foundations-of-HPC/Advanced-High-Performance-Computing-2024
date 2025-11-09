/* ============================================================================
 * EXAMPLE 2: Loop-Carried Dependency (RAW - Read After Write)
 * ============================================================================
 * 
 * PURPOSE: Demonstrate why certain loops CANNOT be vectorized due to
 *          true data dependencies between iterations.
 *
 * LEARNING GOALS:
 * - Recognize "loop not vectorized: loop contains data dependencies"
 * - Understand what RAW (Read-After-Write) dependency means
 * - Learn when dependencies are fundamental vs. fixable
 *
 * COMPILE WITH:
 *   gcc -O3 -march=native -ftree-vectorize -fopt-info-vec-all 02_dependency_failure.c -o dep_fail
 *
 * EXPECTED OUTPUT (GCC):
 *   02_dependency_failure.c:38:5: missed: couldn't vectorize loop
 *   02_dependency_failure.c:38:5: missed: not vectorized: loop contains data dependencies.
 *   02_dependency_failure.c:39:9: missed: statement clobbers memory: a[i_23] = _4;
 *
 * KEY OBSERVATIONS:
 * - "missed: couldn't vectorize loop" = FAILURE
 * - "data dependencies" = iteration i depends on iteration i-1
 * - This is a REAL dependency - cannot be eliminated
 *
 * WHY IT FAILS:
 *   a[i] = a[i-1] + c[i]
 *        ^         ^
 *        WRITE     READ from previous iteration
 *
 * This creates a dependency chain: must compute a[0] before a[1], a[1] before a[2], etc.
 * Vectorization requires INDEPENDENCE - processing multiple iterations simultaneously.
 *
 * Dependency chain
 *
 * a[0] <- c[0]
 *  \_______
 *          \
 * a[1] <- a[0] + c[1]
 *   \_______
 *          \
 * a[2] <- a[1] + c[2]
 *   \_______
 *          \
 * a[3] <- a[2] + c[2]
 *
 * ....
 *
 * ============================================================================
 */

#include <stdlib.h>
#include <stdio.h>

#define N 1000000

/*
 *  HINT : try to use `ivdep` directive, and check that you get
 *         a vector code that produces garbage
 */


// Cumulative sum - classic example of loop-carried dependency
void cumulative_sum(double * restrict a, const double * restrict c, int n)
{
    a[0] = c[0];
    for (int i = 1; i < n; i++) {
        a[i] = a[i-1] + c[i];  // READ a[i-1], WRITE a[i]
                               // Iteration i depends on iteration i-1
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : N;
    
    double *a = (double*)malloc(n * sizeof(double));
    double *c = (double*)malloc(n * sizeof(double));
    
    // Initialize
    for (int i = 0; i < n; i++) {
        c[i] = 1.0;  // Simple values for testing
    }
    
    // Execute
    cumulative_sum(a, c, n);
    
    // Verify (should be: a[i] = i+1)
    printf("Result: a[0] = %g, a[999] = %g (expected 1000)\n", a[0], a[999]);
    
    free(c); free(a);
    return 0;
}

/* ============================================================================
 * TEACHER NOTES:
 * 
 * This is a FUNDAMENTAL dependency that cannot be fixed by:
 * - Adding #pragma GCC ivdep (would produce wrong results!)
 * - Using restrict keyword
 * - Rearranging code
 *
 * The algorithm itself is inherently sequential.
 *
 * However, for REDUCTIONS (sum, max, min), we can manually unroll with
 * multiple accumulators. That's covered in the "Loop Dependencies" module.
 *
 * For CUMULATIVE operations like this, vectorization is generally not possible.
 * Some advanced techniques exist (prefix sum algorithms on GPU), but they're
 * algorithmic changes, not compiler tricks.
 * ============================================================================
 */
