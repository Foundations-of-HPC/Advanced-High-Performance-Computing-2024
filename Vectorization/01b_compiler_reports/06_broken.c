/* ============================================================================
 * DIAGNOSTIC EXERCISE: Pointer Aliasing Prevents Vectorization
 * ============================================================================
 * 
 * SCENARIO: You've written a simple array scaling function, but it runs slow.
 * You suspect vectorization isn't happening. Use compiler reports to diagnose!
 *
 * ASSIGNMENT:
 * 1. Compile with: gcc -O3 -march=native -fopt-info-vec-all 06_broken_aliasing.c -o broken
 * 2. Read the compiler output - what does it say?
 * 3. Identify WHY vectorization failed
 * 4. Fix the code
 * 5. Verify the fix worked
 *
 * ============================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../headers/timing.h"

#define N 10000000

/* ============================================================================
 * BROKEN VERSION - Does NOT vectorize
 * ============================================================================
 */

// Scale array by a factor
void scale_array_broken(double *output, double *input, double factor, int n)
{
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * factor;
    }
}

/* ============================================================================
 * WHAT WILL THE COMPILER REPORT SAY?
 * ============================================================================
 * 
 * Compile this and check:
 *   gcc -O3 -march=native -fopt-info-vec-all 06_broken_aliasing.c -c
 * 
 * EXPECTED OUTPUT:
 *   06_broken_aliasing.c:23:5: missed: couldn't vectorize loop
 *   06_broken_aliasing.c:23:5: missed: not vectorized: loop contains data 
 *                                      dependencies or possible aliasing.
 * 
 * KEY PHRASE: "possible aliasing"
 * 
 * WHAT DOES THIS MEAN?
 * ============================================================================
 * 
 * ALIASING = Two pointers might point to overlapping memory regions
 * 
 * The compiler doesn't know if 'output' and 'input' point to:
 * - Completely separate arrays (no aliasing) → Safe to vectorize
 * - Overlapping arrays (aliasing) → NOT safe to vectorize
 * 
 * Example of DANGEROUS overlap:
 *   double array[100];
 *   scale_array_broken(array + 1, array, 2.0, 99);
 *   // output[0] = array[1], input[0] = array[0]
 *   // output[1] = array[2], input[1] = array[1]  <-- Reading output[0]!
 * 
 * If vectorized, we'd read multiple input[] values simultaneously,
 * but we might be overwriting input[] at the same time!
 * 
 * COMPILER MUST BE CONSERVATIVE:
 * Without proof that pointers don't overlap, can't vectorize.
 * 
 * ============================================================================
 */

/* ============================================================================
 * TEST HARNESS
 * ============================================================================
 */

int main(int argc, char **argv)
{
    int n = (argc > 1) ? atoi(argv[1]) : N;
    
    double *input = (double*)malloc(n * sizeof(double));
    double *output = (double*)malloc(n * sizeof(double));
    
    if (!input || !output) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    
    // Initialize
    for (int i = 0; i < n; i++) {
        input[i] = (double)i;
    }
    
    // Warm up
    scale_array_broken(output, input, 2.0, n);
    
    // Time it
    double elapsed = PCPU_TIME;
    for (int rep = 0; rep < 10; rep++) {
        scale_array_broken(output, input, 2.0, n);
    }
    elapsed = PCPU_TIME - elapsed;
    
    printf("BROKEN version:\n");
    printf("  Time: %.3f ms per iteration\n", elapsed);
    printf("  Result check: output[0] = %g, output[%d] = %g\n", 
           output[0], n-1, output[n-1]);
    
    free(output);
    free(input);
    
    return 0;
}

/* ============================================================================
 * DIAGNOSTIC QUESTIONS:
 * ============================================================================
 * 
 * After compiling with -fopt-info-vec-all, answer these:
 * 
 * 1. Did the loop vectorize?
 *    → Check compiler output for "optimized: loop vectorized" (success)
 *      or "missed: not vectorized" (failure)
 * 
 * 2. If it didn't vectorize, what was the reason?
 *    → Look for the specific failure message
 *    → In this case: "possible aliasing" or "data dependencies"
 * 
 * 3. What does "possible aliasing" mean?
 *    → Compiler can't prove pointers don't overlap
 *    → Must assume they MIGHT overlap → can't vectorize
 * 
 * 4. How can we fix this?
 *    → See the FIXED version below!
 * 
 * ============================================================================
 */
