/*
 * Example 3: Potential (False) Dependencies and How to Resolve Them
 * 
 * Demonstrates:
 * - Pointer aliasing preventing vectorization
 * - Array indices creating potential dependencies
 * - How to use 'restrict' keyword
 * - How to use #pragma ivdep / #pragma GCC ivdep
 * - When these optimizations are SAFE vs UNSAFE
 *
 * Compile with:
 * gcc -O3 -march=native -fopt-info-vec-all 03_potential_dependencies.c -o potential_dependencies
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../headers/timing.h>

#define N 10000000

/* ========================================
 * EXAMPLE 1: Pointer Aliasing Problem
 * ======================================== */

// PROBLEM: Compiler doesn't know if 'a' and 'b' overlap
void vector_add_aliasing(double *a, double *b, double *c, int n)
{
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];  // Might not vectorize due to potential aliasing
}

// SOLUTION 1: Use 'restrict' keyword - Promise no aliasing
void vector_add_restrict(double * restrict a, double * restrict b, 
                         double * restrict c, int n)
{
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];  // Will vectorize - no aliasing possible
}

// SOLUTION 2: Use compiler pragma
void vector_add_pragma(double *a, double *b, double *c, int n)
{
 #pragma GCC ivdep  // Tell GCC to ignore vector dependencies
  for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];    
}

/* ========================================
 * EXAMPLE 2: Self-Assignment with Offset (DANGEROUS!)
 * ======================================== */

// This looks like it might vectorize, but careful!
void shift_array_unsafe(double *array, int n, int offset)
{
  // DANGER: If offset < vector_width, we have overlap!
  for (int i = 0; i < n - offset; i++)
    array[i] = array[i + offset];  // Read array[i+offset], write array[i]
}

// If offset >= vector width, this is safe to vectorize
void shift_array_safe(double *array, int n, int offset)
{
  if (offset >= 4)
    {  // Safe for AVX (256-bit = 4 doubles)
     #pragma omp simd
      for (int i = 0; i < n - offset; i++)
	array[i] = array[i + offset];
    }
  else
    {
      // Fall back to scalar for small offsets
      for (int i = 0; i < n - offset; i++)
	array[i] = array[i + offset];
    }
}

/* ========================================
 * EXAMPLE 3: Indirect Addressing (Gather/Scatter)
 * ======================================== */

// PROBLEM: Compiler can't prove indices don't overlap
void indirect_sum_unsafe(double *a, double *b, int *indices, int n)
{
    for (int i = 0; i < n; i++)
      a[indices[i]] += b[i];  // Potential write conflict if indices overlap
}

// If you KNOW indices are unique, you can assert it
void indirect_sum_safe(double *a, double *b, int *indices, int n)
{
 #pragma GCC ivdep  // Use ONLY if you're CERTAIN indices don't repeat!
  for (int i = 0; i < n; i++) {
    a[indices[i]] += b[i];
}

// SAFER: Read-only indirect access (gather) is always safe
void indirect_gather(double *a, double *b, int *indices, int n)
{
 #pragma omp simd  // Safe - we're only reading a[], writes to b[] are contiguous
  for (int i = 0; i < n; i++)
    b[i] = a[indices[i]];  // Gather operation - safe to vectorize
}

/* ========================================
 * EXAMPLE 4: Loop-Carried Anti-Dependency (WAR)
 * ======================================== */

// Read x[i+1], write x[i] - looks dangerous but might be safe
void war_potential(double *x, int n)
{
  for (int i = 0; i < n - 1; i++)
    x[i] = x[i+1] * 2.0;  // Read ahead by 1
    
  // For vectorization: if we process 4 elements at once:
  // Iteration 0-3: write x[0-3], read x[1-4]
  // Problem: iteration 0 writes x[0], iteration -1 (if it existed) would read x[0]
  // But since we're reading AHEAD, not behind, this is actually SAFE
}

// The compiler should vectorize the above, but might be conservative
// We can help it:
void war_potential_explicit(double *x, int n)
{
 #pragma GCC ivdep
  for (int i = 0; i < n - 1; i++)
    x[i] = x[i+1] * 2.0;
}

/* ========================================
 * EXAMPLE 5: Conditional Dependencies
 * ======================================== */

// Compiler might worry about the conditional
 void conditional_potential(double *a, double *b, int n)
 {
   for (int i = 0; i < n; i++)
     {
       if (a[i] > 0)
	 b[i] = a[i] * 2.0;
       else
	 b[i] = a[i] * 3.0;
     }
   // This SHOULD vectorize (using blend/mask), but conservative compilers might not
}

// Make it explicit with ternary operator (often helps)
void conditional_ternary(double *a, double *b, int n)
{
 #pragma omp simd
  for (int i = 0; i < n; i++)
    b[i] = (a[i] > 0) ? (a[i] * 2.0) : (a[i] * 3.0);
}

/* ========================================
 * EXAMPLE 6: When restrict is WRONG and DANGEROUS
 * ======================================== */

// WRONG: These pointers DO alias!
void wrong_restrict_example(double * restrict x, double * restrict y, int n)
{
  for (int i = 0; i < n - 1; i++)
    x[i] = y[i] + y[i+1];
}

void demonstrate_wrong_restrict()
{
  double array[100];
  for (int i = 0; i < 100; i++) array[i] = i;
  
  // DISASTER: Passing same array as both arguments!
  // But we promised with 'restrict' they don't alias
  wrong_restrict_example(array, array, 99);
  
  // Result is UNDEFINED BEHAVIOR - might work, might not, might crash
  printf("Result (undefined behavior): ");
  for (int i = 0; i < 10; i++) printf("%.1f ", array[i]);
  printf("\n");
}

/* ========================================
 * Timing Tests
 * ======================================== */

int main( void )
{
  double *a = (double*)aligned_alloc(64, N * sizeof(double));
  double *b = (double*)aligned_alloc(64, N * sizeof(double));
  double *c = (double*)aligned_alloc(64, N * sizeof(double));
    
  // Initialize
  for (int i = 0; i < N; i++)
    {
      a[i] = (double)i;
      b[i] = (double)(i * 2);
    }
    
  printf("=== POINTER ALIASING PERFORMANCE ===\n\n");
    
  // Test 1: Without restrict
  double t0 = PCPU_TIME;
  vector_add_aliasing(a, b, c, N);
  double t1 = PCPU_TIME;
  printf("Without restrict:  %.3f ms\n", (t1-t0)*1000);
    
  // Test 2: With restrict
  t0 = PCPU_TIME;
  vector_add_restrict(a, b, c, N);
  t1 = PCPU_TIME;
  printf("With restrict:     %.3f ms (speedup: %.2fx)\n", 
	 (t1-t0)*1000, (t1-t0) > 0 ? ((double)(t1-t0))/(t1-t0) : 0);
    
  // Test 3: With pragma
  t0 = PCPU_TIME;
  vector_add_pragma(a, b, c, N);
  t1 = PCPU_TIME;
  printf("With #pragma:      %.3f ms\n\n", (t1-t0)*1000);
    
  printf("=== SAFETY RULES FOR VECTORIZATION HINTS ===\n\n");
    
  printf("[Y] SAFE to use 'restrict' when:\n");
  printf("    • Different malloc'd arrays\n");
  printf("    • Arrays from different function arguments\n");
  printf("    • You control all callers and ensure no aliasing\n\n");
    
  printf("[Y] SAFE to use #pragma ivdep when:\n");
  printf("    • Reading ahead (WAR with positive offset)\n");
  printf("    • Gather operations (reading via indices)\n");
  printf("    • You've verified no loop-carried dependencies\n\n");
    
  printf("[X] UNSAFE (undefined behavior) when:\n");
  printf("    • Pointers might actually overlap\n");
  printf("    • True loop-carried dependencies exist\n");
  printf("    • Scatter with potentially duplicate indices\n");
  printf("    • Backward dependencies (reading x[i-k])\n\n");
    
  printf("=== COMPILER REPORT KEYWORDS ===\n");
  printf("Look for these in -fopt-info-vec-all:\n");
  printf("• 'possible aliasing' → Try restrict\n");
  printf("• 'data dependencies' → Check if true or false dependency\n");
  printf("• 'not vectorized' → Investigate why\n");
  printf("• 'versioned' → Compiler made two versions (with/without vectorization)\n\n");
    
  printf("=== BEST PRACTICES ===\n");
  printf("1. Design APIs to avoid aliasing (separate input/output)\n");
  printf("2. Use restrict judiciously - only when truly safe\n");
  printf("3. Verify with compiler reports AND runtime testing\n");
  printf("4. When in doubt, let compiler be conservative\n");
  printf("5. Premature optimization: forcing vectorization unsafely is a bug\n");
    
  free(a);
  free(b);
  free(c);
    
  return 0;
}
