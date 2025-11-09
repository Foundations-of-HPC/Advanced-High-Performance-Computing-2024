/*
 * Example 1: Reduction Patterns and Loop-Carried Dependencies
 * 
 * Demonstrates:
 * - True loop-carried dependency (sum reduction)
 * - How compilers handle reductions
 * - OpenMP reduction clause for vectorization
 * - Manual unrolling to expose parallelism
 *
 * Compile with:
 * gcc -O3 -march=native -fopt-info-vec-all 01_reduction_patterns.c -o reduction_patterns
 */

#include <stdio.h>
#include <stdlib.h>
#include "../../headers/timing.h"


#define N 100000000

/* ========================================
 * EXAMPLE 1A: Simple Sum - Loop-Carried Dependency
 * ======================================== */

// This has a true loop-carried dependency
// Each iteration depends on the previous: sum[i] = sum[i-1] + array[i]
double sum_scalar(double *array, int n)
{
  double sum = 0.0;
  for (int i = 0; i < n; i++)
    sum += array[i];  // Read-after-write dependency across iterations
    
  return sum;
}

/* ========================================
 * EXAMPLE 1B: Compiler Recognizes Reduction
 * ======================================== */

// Modern compilers recognize reduction patterns and can vectorize them
// using horizontal addition (vpaddpd, haddpd, etc.)
double sum_auto_vectorized(double *array, int n)
{
  double sum = 0.0;
 #pragma GCC ivdep  // Hint: no other dependencies
  for (int i = 0; i < n; i++)
    sum += array[i];
  return sum;
}

/* ========================================
 * EXAMPLE 1C: Manual Unrolling with Multiple Accumulators
 * ======================================== */

// Break the dependency chain by using multiple independent accumulators
// This exposes more parallelism for the compiler
double sum_unrolled_4way(double *array, int n)
{
  double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
  
  int i;
  for (i = 0; i < n - 3; i += 4)
    {
      sum0 += array[i + 0];
      sum1 += array[i + 1];
      sum2 += array[i + 2];
      sum3 += array[i + 3];
    }
  
  // Handle remainder
  double sum = sum0 + sum1 + sum2 + sum3;
  for (; i < n; i++)
    sum += array[i];
    
  return sum;
}

/* ========================================
 * EXAMPLE 1D: OpenMP SIMD Reduction
 * ======================================== */

// OpenMP provides explicit reduction clause
double sum_omp_simd(double *array, int n)
{
  double sum = 0.0;
 #pragma omp simd reduction(+:sum)
  for (int i = 0; i < n; i++)
    sum += array[i];

  return sum;
}

/* ========================================
 * EXAMPLE 2: Max Finding - Another Reduction
 * ======================================== */

// Maximum finding is also a reduction operation
double max_scalar(double *array, int n)
{
  double max_val = array[0];
  for (int i = 1; i < n; i++)
    {
      if (array[i] > max_val) 
	max_val = array[i];
    }
  return max_val;
}

// Compiler can vectorize max reductions using vector max instructions (vmaxpd)
double max_vectorized(double *array, int n)
{
  double max_val = array[0];
 #pragma omp simd reduction(max:max_val)
  for (int i = 1; i < n; i++)
    if (array[i] > max_val)
      max_val = array[i];
    
  return max_val;
}

/* ========================================
 * EXAMPLE 3: Dot Product - Combination Pattern
 * ======================================== */

double dot_product(double *a, double *b, int n)
{
  double sum = 0.0;
 #pragma omp simd reduction(+:sum)
  for (int i = 0; i < n; i++)
    sum += a[i] * b[i];  // Multiply-add pattern, perfect for FMA
  return sum;
}

/* ========================================
 * Main: Timing and Verification
 * ======================================== */

/*
double get_time()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}
*/
int main()
{
  double *array = (double*)aligned_alloc(64, N * sizeof(double));
  double *array2 = (double*)aligned_alloc(64, N * sizeof(double));

  srand48(time(NULL));
  // Initialize with random values
  for (int i = 0; i < N; i++)
    {
      array[i] = (double)drand48();
      array2[i] = (double)drand48();
    }
    
  printf("Testing with %d elements\n\n", N);
  
  // Test sum operations
  printf("=== SUM REDUCTION ===\n");
  
  double t0 = PCPU_TIME;
  double sum1 = sum_scalar(array, N);
  double t1 = PCPU_TIME;
  printf("Scalar sum:           %.6f (%.3f ms)\n", sum1, (t1-t0)*1000);
  
  t0 = PCPU_TIME;
  double sum2 = sum_auto_vectorized(array, N);
  t1 = PCPU_TIME;
  printf("Auto-vectorized sum:  %.6f (%.3f ms) - speedup: %.2fx\n", 
	 sum2, (t1-t0)*1000, (t1-t0) > 0 ? ((double)(t1-t0))/(t1-t0) : 0);
  
  t0 = PCPU_TIME;
  double sum3 = sum_unrolled_4way(array, N);
  t1 = PCPU_TIME;
  printf("4-way unrolled sum:   %.6f (%.3f ms)\n", sum3, (t1-t0)*1000);
  
  t0 = PCPU_TIME;
  double sum4 = sum_omp_simd(array, N);
  t1 = PCPU_TIME;
  printf("OpenMP SIMD sum:      %.6f (%.3f ms)\n", sum4, (t1-t0)*1000);
  
  // Test max operation
  printf("\n=== MAX REDUCTION ===\n");
  
  t0 = PCPU_TIME;
  double max1 = max_scalar(array, N);
  t1 = PCPU_TIME;
  printf("Scalar max:           %.6f (%.3f ms)\n", max1, (t1-t0)*1000);
  
  t0 = PCPU_TIME;
  double max2 = max_vectorized(array, N);
  t1 = PCPU_TIME;
  printf("Vectorized max:       %.6f (%.3f ms)\n", max2, (t1-t0)*1000);
  
  // Test dot product
  printf("\n=== DOT PRODUCT ===\n");
  
  t0 = PCPU_TIME;
  double dot = dot_product(array, array2, N);
  t1 = PCPU_TIME;
  printf("Dot product:          %.6f (%.3f ms)\n", dot, (t1-t0)*1000);
  
  printf("\n=== KEY OBSERVATIONS ===\n");
  printf("1. Compilers recognize reduction patterns\n");
  printf("2. OpenMP reduction clause is most portable\n");
  printf("3. Manual unrolling can help but not always necessary\n");
  printf("4. Watch for FMA opportunities (multiply-add patterns)\n");
  
  free(array);
  free(array2);
  return 0;
}
