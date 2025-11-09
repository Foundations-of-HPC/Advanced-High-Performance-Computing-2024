/* ============================================================================
 * EXAMPLE 4: Alignment Uncertainty
 * ============================================================================
 * 
 * PURPOSE: Show how the compiler's inability to prove alignment affects
 *          vectorization decisions and code generation.
 *
 * LEARNING GOALS:
 * - Understand "loop versioned for vectorization" vs. "loop peeled"
 * - Learn how to communicate alignment guarantees to compiler
 * - See performance impact of alignment assumptions
 *
 * COMPILE WITH:
 *   gcc -O3 -march=native -ftree-vectorize -fopt-info-vec-all 04_alignment_uncertainty.c -o align
 *
 * EXPECTED OUTPUT (GCC, without __builtin_assume_aligned):
 *   04_alignment_uncertainty.c:48:5: optimized: loop vectorized using 32 byte vectors
 *   04_alignment_uncertainty.c:48:5: optimized: loop versioned for vectorization because of possible aliasing
 *   04_alignment_uncertainty.c:48:5: optimized: loop peeled for vectorization to enhance alignment
 *
 * KEY OBSERVATIONS:
 * - "loop versioned" = compiler creates TWO versions (aligned + unaligned paths)
 * - "loop peeled" = scalar prologue to reach alignment boundary
 * - Runtime checks add overhead
 *
 * WITH __builtin_assume_aligned:
 *   04_alignment_uncertainty.c:48:5: optimized: loop vectorized using 32 byte vectors
 *   [No versioning or peeling messages]
 *
 * - Simpler code, no runtime checks
 * - Compiler trusts your alignment guarantee
 *
 * ============================================================================
 */

#include <stdlib.h>
#include <stdio.h>

#define N 1000000
#define ALIGN 32  // AVX2 alignment

// Version 1: Compiler doesn't know alignment
void vector_add_unknown_align(double * restrict a, 
                               const double * restrict b,
                               const double * restrict c, int n)
{
    for (int i = 0; i < n; i++)
        a[i] = b[i] + c[i];   
}

// Version 2: Tell compiler about alignment
void vector_add_known_align(double * restrict a, 
                             const double * restrict b,
                             const double * restrict c, int n)
{
    // Communicate alignment to compiler
    a = (double*)__builtin_assume_aligned(a, ALIGN);
    b = (const double*)__builtin_assume_aligned(b, ALIGN);
    c = (const double*)__builtin_assume_aligned(c, ALIGN);
    
    for (int i = 0; i < n; i++)
        a[i] = b[i] + c[i];
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : N;
    
    // Allocate aligned memory
    double *a = (double*)aligned_alloc(ALIGN, n * sizeof(double));
    double *b = (double*)aligned_alloc(ALIGN, n * sizeof(double));
    double *c = (double*)aligned_alloc(ALIGN, n * sizeof(double));
    
    if (!a || !b || !c) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    
    // Initialize
    for (int i = 0; i < n; i++)
      {
        b[i] = (double)i;
        c[i] = (double)(i * 2);
      }
    
    // Test both versions
    #ifdef USE_ASSUME_ALIGNED
    vector_add_known_align(a, b, c, n);
    printf("Using __builtin_assume_aligned version\n");
    #else
    vector_add_unknown_align(a, b, c, n);
    printf("Using unknown alignment version\n");
    #endif
    
    printf("Result: a[0] = %g, a[%d] = %g\n", a[0], n-1, a[n-1]);
    
    free(c); free(b); free(a);
    return 0;
}

/* ============================================================================
 * PERFORMANCE TEST:
 * 
 * Compile both versions and time them:
 * 
 * gcc -O3 -march=native 04_alignment_uncertainty.c -o align_unknown
 * gcc -O3 -march=native -DUSE_ASSUME_ALIGNED 04_alignment_uncertainty.c -o align_known
 * 
 * time ./align_unknown 100000000
 * time ./align_known 100000000
 * 
 * Expected: align_known is 5-15% faster due to:
 * - No runtime alignment checks
 * - No loop versioning overhead
 * - Simpler generated code
 * - Better instruction scheduling
 *
 * ============================================================================
 * TEACHER NOTES:
 * 
 * When to use __builtin_assume_aligned:
 * 
 * 1. YOU control memory allocation (via aligned_alloc, posix_memalign)
 * 2. Data comes from aligned arrays (declared with __attribute__((aligned(N))))
 * 3. You've manually verified alignment (assert((uintptr_t)ptr % ALIGN == 0))
 *
 * When NOT to use:
 * 
 * 1. Memory from unknown source (function parameters from external code)
 * 2. malloc() without verification (malloc alignment is implementation-dependent)
 * 3. Middle of arrays (ptr+offset might not be aligned even if ptr is)
 *
 * Consequences of lying:
 * 
 * If you tell the compiler "this is aligned" and it's NOT, you may get:
 * - Segmentation faults (on some older CPUs)
 * - Silent data corruption
 * - Incorrect results
 * - Performance degradation
 *
 * The compiler TRUSTS you. Use responsibly.
 * ============================================================================
 */
