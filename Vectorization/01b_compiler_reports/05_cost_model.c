/* ============================================================================
 * EXAMPLE 5: Cost Model Rejection ("Not Profitable")
 * ============================================================================
 * 
 * PURPOSE: Demonstrate cases where the compiler CAN vectorize but chooses
 *          NOT to because its cost model predicts slower performance.
 *
 * LEARNING GOALS:
 * - Understand "not vectorized: not profitable"
 * - Learn what factors make vectorization unprofitable
 * - Know when to override cost model vs. trust it
 *
 * COMPILE WITH:
 *   gcc -O3 -march=native -ftree-vectorize -fopt-info-vec-all 05_cost_model.c -o cost
 *
 * EXPECTED OUTPUT (GCC):
 *   05_cost_model.c:44:5: missed: couldn't vectorize loop
 *   05_cost_model.c:44:5: missed: not vectorized: vectorization not profitable.
 *
 * KEY OBSERVATIONS:
 * - Compiler CAN vectorize, but predicts it would be SLOWER
 * - Short loop count is main culprit
 * - Startup overhead > benefit from vectorization
 *
 * ============================================================================
 */

#include <stdlib.h>
#include <stdio.h>

// Very short loop - vectorization overhead may exceed benefit
void short_loop_add(double *a, const double *b, const double *c)
{
    // Only 8 iterations - cost model likely rejects vectorization
    for (int i = 0; i < 8; i++) {
        a[i] = b[i] + c[i];
    }
}

// Complex loop body - cost model considers total work
void complex_body(double *result, const double *data, int n)
{
    for (int i = 0; i < n; i++) {
        // Multiple scalar operations
        double x = data[i];
        double temp = x * x;           // multiply
        temp = temp + x;               // add
        temp = temp / (x + 1.0);       // divide
        temp = temp - 0.5;             // subtract
        result[i] = temp;
    }
}

/* ============================================================================
 * WHY VECTORIZATION MIGHT NOT BE PROFITABLE:
 * 
 * 1. SHORT LOOPS:
 *    - Vectorization setup overhead (loading constants, setting up masks)
 *    - Loop prologue/epilogue for alignment
 *    - Scalar remainder iterations
 *    - For <16 iterations, overhead often exceeds benefit
 *
 * 2. MEMORY-BOUND CODE:
 *    - If bottleneck is memory bandwidth, not compute
 *    - Vector loads don't help if waiting for DRAM
 *    - Example: streaming large arrays once
 *
 * 3. MIXED SCALAR/VECTOR:
 *    - If loop contains operations that don't vectorize well
 *    - Division, square root, transcendentals (expensive even vectorized)
 *    - Horizontal operations (reductions within vector)
 *
 * 4. SMALL DATA TYPES WITH LARGE VECTORS:
 *    - AVX-512 with 8-bit data = 64 elements/vector
 *    - If loop count < 64, lots of wasted lanes
 *
 * ============================================================================
 */

/* ============================================================================
 * FORCING VECTORIZATION:
 * 
 * You can override the cost model with pragmas, but be careful!
 * ============================================================================
 */

void short_loop_forced(double *a, const double *b, const double *c)
{
    // Force vectorization even if cost model says no
    #pragma GCC ivdep
    #pragma GCC optimize("tree-vectorize")
    for (int i = 0; i < 8; i++) {
        a[i] = b[i] + c[i];
    }
}

/* ============================================================================
 * COMPILE WITH FORCED VECTORIZATION:
 *   gcc -O3 -march=native -fopt-info-vec-all 05_cost_model.c \
 *       -DFORCE_VEC -o cost_forced
 *
 * The compiler may still reject! But you can try:
 *   gcc -O3 -march=native -fopt-info-vec-all 05_cost_model.c \
 *       -ftree-vectorizer-verbose=2 -o cost_verbose
 *
 * This gives MORE details about why cost model rejected it.
 * ============================================================================
 */

int main() {
    double a[8], b[8], c[8];
    
    // Initialize
    for (int i = 0; i < 8; i++) {
        b[i] = (double)i;
        c[i] = (double)(i * 2);
    }
    
    #ifdef FORCE_VEC
    short_loop_forced(a, b, c);
    printf("Forced vectorization version\n");
    #else
    short_loop_add(a, b, c);
    printf("Normal (cost model decides) version\n");
    #endif
    
    printf("Result: a[0] = %g, a[7] = %g\n", a[0], a[7]);
    
    // Test complex body
    double data[100], result[100];
    for (int i = 0; i < 100; i++) data[i] = (double)(i+1);
    complex_body(result, data, 100);
    printf("Complex body: result[0] = %g\n", result[0]);
    
    return 0;
}

/* ============================================================================
 * TEACHER NOTES:
 * 
 * When to TRUST the cost model:
 * - You're writing general-purpose library code
 * - You don't know the target CPU microarchitecture
 * - Loop count is variable/unknown at compile time
 *
 * When to OVERRIDE the cost model:
 * - You've MEASURED and vectorized version is actually faster
 * - You know loop count will be large at runtime (despite small constant in code)
 * - You're targeting specific CPU and know its characteristics
 * - Profiling shows this loop is a hotspot
 *
 * How to MEASURE:
 * - Use perf counters: cycles_per_instruction, vector_instructions_retired
 * - Use rdtsc or clock_gettime for timing
 * - Test with realistic data and loop counts
 * - Compare multiple input sizes
 *
 * The cost model is generally good, but it's not perfect:
 * - Based on heuristics and microarchitecture models
 * - May not account for your specific cache usage patterns
 * - Assumes "typical" data and doesn't know your use case
 *
 * Rule of thumb:
 * - Loop count < 16: Vectorization rarely helps
 * - Loop count 16-64: Cost model is crucial, measure carefully
 * - Loop count > 64: Vectorization usually helps (if no dependencies)
 * ============================================================================
 */
