/*
 * Example 2: True Loop-Carried Dependencies That Prevent Vectorization
 * 
 * Demonstrates:
 * - Recurrence relations (Fibonacci-like)
 * - Forward dependencies (each iteration uses previous result)
 * - Backward dependencies (write-after-read)
 * - Why these CANNOT be vectorized
 * - Techniques to work around them when possible
 *
 * Compile with:
 * gcc -O3 -march=native -fopt-info-vec-all 02_true_dependencies.c -o true_dependencies
 *
 * EXPECTED: Compiler will report vectorization failures for these loops
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1000000

/* ========================================
 * EXAMPLE 1: Recurrence Relation (CANNOT VECTORIZE)
 * ======================================== */

// Classic Fibonacci-like recurrence: x[i] = x[i-1] + x[i-2]
// Each iteration MUST wait for previous two iterations
void fibonacci_recurrence(double *x, int n)
{
  x[0] = 1.0;
  x[1] = 1.0;
  
  for (int i = 2; i < n; i++)
    x[i] = x[i-1] + x[i-2];  // True dependency - CANNOT vectorize

}

/* ========================================
 * EXAMPLE 2: Forward Dependency Chain (CANNOT VECTORIZE)
 * ======================================== */

// Each element depends on the computation of the previous element
void forward_dependency(double *input, double *output, int n)
{
  output[0] = input[0];
  
  for (int i = 1; i < n; i++) 
    // output[i] depends on output[i-1] which was computed in previous iteration
    output[i] = output[i-1] * 0.5 + input[i];  // CANNOT vectorize

}

// This is essentially an IIR (Infinite Impulse Response) filter
// y[n] = a * y[n-1] + b * x[n]
void iir_filter(double *input, double *output, int n, double a, double b)
{
  output[0] = b * input[0];
  
  for (int i = 1; i < n; i++)
    output[i] = a * output[i-1] + b * input[i];  // CANNOT vectorize

}

/* ========================================
 * EXAMPLE 3: Prefix Sum / Cumulative Sum (CANNOT VECTORIZE NAIVELY)
 * ======================================== */

// Cumulative sum: output[i] = output[i-1] + input[i]
void prefix_sum_serial(double *input, double *output, int n)
{
    output[0] = input[0];
    
    for (int i = 1; i < n; i++)
      output[i] = output[i-1] + input[i];  // Loop-carried dependency
}

// ALTERNATIVE: Blocked prefix sum (partially vectorizable)
// Divide into blocks, compute block sums vectorized, then fix up
void prefix_sum_blocked(double *input, double *output, int n)
{
  const int BLOCK_SIZE = 8;
  int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  
  double block_sums[num_blocks];
  
  // Step 1: Compute independent block sums (CAN vectorize within blocks)
  for (int b = 0; b < num_blocks; b++)
    {
      int start = b * BLOCK_SIZE;
      int end = (start + BLOCK_SIZE < n) ? start + BLOCK_SIZE : n;
      
      block_sums[b] = 0.0;
      for (int i = start; i < end; i++)
	block_sums[b] += input[i];
    }
  
    // Step 2: Prefix sum of block sums (small, serial)
  for (int b = 1; b < num_blocks; b++)
    block_sums[b] += block_sums[b-1];
    
    // Step 3: Compute final prefix sums within each block
    for (int b = 0; b < num_blocks; b++)
      {
        int start = b * BLOCK_SIZE;
        int end = (start + BLOCK_SIZE < n) ? start + BLOCK_SIZE : n;
        
        double base = (b > 0) ? block_sums[b-1] : 0.0;
        output[start] = base + input[start];
        
        for (int i = start + 1; i < end; i++)
	  output[i] = output[i-1] + input[i];
    }
}

/* ========================================
 * EXAMPLE 4: Loop-Carried Dependency with Distance > 1
 * ======================================== */

// Distance-2 dependency: x[i] depends on x[i-2]
// This CAN sometimes be vectorized by processing even/odd indices separately
void distance_2_dependency(double *x, int n)
{
  for (int i = 2; i < n; i++)
    x[i] = x[i-2] * 0.9 + 1.0;  // Distance-2 dependency
}

// BETTER: Explicitly separate even/odd (easier for compiler)
void distance_2_separated(double *x, int n)
{
  // Process even indices
  for (int i = 2; i < n; i += 2)
    x[i] = x[i-2] * 0.9 + 1.0;  // Can vectorize this

    
    // Process odd indices
    for (int i = 3; i < n; i += 2)
      x[i] = x[i-2] * 0.9 + 1.0;  // Can vectorize this
}

/* ========================================
 * EXAMPLE 5: Write-After-Read (WAR) Dependency
 * ======================================== */

// Classic WAR pattern
void war_dependency(double *x, int n)
{
  for (int i = 0; i < n - 1; i++)
    x[i] = x[i+1] * 2.0;  // Read x[i+1], write x[i]
  // Problem: Next iteration reads x[i+1], but we just wrote to x[i]
  // If i+1 overlaps with next iteration's i, we have a problem
}

// SOLUTION: Use separate input/output arrays
void war_fixed(double *input, double *output, int n)
{
 #pragma omp simd
    for (int i = 0; i < n - 1; i++)
      output[i] = input[i+1] * 2.0;  // No overlap - can vectorize!
}

/* ========================================
 * EXAMPLE 6: Conditional Recurrence (CANNOT VECTORIZE)
 * ======================================== */

// Running maximum with conditional update
void running_max(double *input, double *output, int n)
{
  output[0] = input[0];
  
  for (int i = 1; i < n; i++)
    // Depends on output[i-1], which depends on previous iterations
    output[i] = (input[i] > output[i-1]) ? input[i] : output[i-1];
}

/* ========================================
 * Main: Demonstration and Compiler Report Analysis
 * ======================================== */

int main( void )
{
  double *x = (double*)malloc(N * sizeof(double));
  double *y = (double*)malloc(N * sizeof(double));
  double *z = (double*)malloc(N * sizeof(double));
  
  // Initialize
  for (int i = 0; i < N; i++)
    {
      x[i] = (double)i;
      y[i] = (double)(i % 100);
    }
    
  printf("=== TRUE LOOP-CARRIED DEPENDENCIES ===\n\n");
  
  printf("1. RECURRENCE RELATIONS\n");
  fibonacci_recurrence(x, 100);  // Small N for display
  printf("   Fibonacci-like: x[i] = x[i-1] + x[i-2]\n");
  printf("   First 10 values: ");
  for (int i = 0; i < 10; i++) printf("%.0f ", x[i]);
  printf("\n");
  printf("   [X] CANNOT vectorize - true dependency chain\n\n");
  
  printf("2. FORWARD DEPENDENCIES (IIR FILTER)\n");
  forward_dependency(y, z, N);
  printf("   Each output depends on previous output\n");
  printf("  [X] CANNOT vectorize - must process sequentially\n\n");
  
  printf("3. PREFIX SUM (CUMULATIVE SUM)\n");
  prefix_sum_serial(y, x, N);
  printf("   Naive approach: [X] CANNOT vectorize\n");
  prefix_sum_blocked(y, z, N);
  printf("   Blocked approach: [Y] Partially vectorizable\n");
  // Verify they match
  int match = 1;
  for (int i = 0; i < N; i++) {
    if (x[i] != z[i]) { match = 0; break; }
  }
  printf("   Results match: %s\n\n", match ? "YES" : "NO");
  
  printf("4. DISTANCE-2 DEPENDENCY\n");
  for (int i = 0; i < N; i++) x[i] = i;
  distance_2_dependency(x, N);
  printf("   x[i] depends on x[i-2]\n");
  printf("   »»  Might vectorize with loop distribution\n\n");
  
  printf("5. WRITE-AFTER-READ (WAR)\n");
  for (int i = 0; i < N; i++) x[i] = i;
  war_dependency(x, N);
  printf("   Reading x[i+1] while writing x[i]\n");
  printf("   Solution: Use separate arrays → ✓ Vectorizable\n\n");
  
  printf("=== KEY TAKEAWAYS ===\n");
  printf("• True dependencies CANNOT be parallelized\n");
  printf("• Look for: x[i] = f(x[i-1], x[i-2], ...)\n");
  printf("• Compiler reports will show 'loop carried data dependence'\n");
  printf("• Sometimes can restructure (blocking, separate passes)\n");
  printf("• When stuck: consider algorithms that avoid recurrence\n\n");
  
  printf("=== WHAT TO LOOK FOR IN COMPILER REPORTS ===\n");
  printf("GCC: 'loop carried data dependence'\n");
  printf("     'not vectorized: loop with data dependencies'\n");
  printf("Clang: 'loop not vectorized: backward data dependencies'\n");
  printf("Intel: 'vector dependence prevents vectorization'\n");
  
  free(x);
  free(y);
  free(z);
  return 0;
}
