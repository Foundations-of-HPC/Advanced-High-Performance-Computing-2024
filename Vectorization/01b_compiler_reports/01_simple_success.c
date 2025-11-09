/* ============================================================================:x

 * EXAMPLE 1: Simple Success Case
 * ============================================================================
 * 
 * PURPOSE: Demonstrate a loop that vectorizes successfully with clear 
 *          compiler reports showing what optimizations were applied.
 *
 * LEARNING GOALS:
 * - Understand what "success" looks like in compiler reports
 * - See the structure of -fopt-info-vec-optimized output
 * - Recognize key phrases: "vectorized", "loop", "iterations"
 *
 * COMPILE WITH:
 *   gcc -O3 -march=native -ftree-vectorize -fopt-info-vec-all 01_simple_success.c -o success
 *
 * EXPECTED OUTPUT (GCC):
 *   01_simple_success.c:32:5: optimized: loop vectorized using 16 byte vectors
 *   01_simple_success.c:32:5: optimized:  loop versioned for vectorization because of possible aliasing
 *
 * KEY OBSERVATIONS:
 * - "optimized: loop vectorized" = SUCCESS!
 * - "16 byte vectors" = SSE (128-bit), "32 byte" = AVX2, "64 byte" = AVX-512
 * - "versioned" = compiler created two versions (vector + scalar fallback)
 *
 *
 *  »»»»»» QUESTIONS
 *  [1] when compiling for x86_64 you should get multiple messages about the loop
 *      in function vector_add being vectorized with different vector sizes.
 *      Why is that so?
 *  [2] you should get the above mentioned group of messages twice. Why is that so?
 * ============================================================================
 */

#include <stdlib.h>
#include <stdio.h>

#define N 1000000

// Simple vector addition - textbook vectorizable loop
void vector_add(double * restrict a, const double * restrict b, 
                const double * restrict c, int n)
{
    for (int i = 0; i < n; i++) {
        a[i] = b[i] + c[i];
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : N;
    
    double *a = (double*)malloc(n * sizeof(double));
    double *b = (double*)malloc(n * sizeof(double));
    double *c = (double*)malloc(n * sizeof(double));
    
    // Initialize
    for (int i = 0; i < n; i++) {
        b[i] = (double)i;
        c[i] = (double)(i * 2);
    }
    
    // Execute
    vector_add(a, b, c, n);
    
    // Prevent dead code elimination
    printf("Result: a[0] = %g, a[%d] = %g\n", a[0], n-1, a[n-1]);
    
    free(c); free(b); free(a);
    return 0;
}
