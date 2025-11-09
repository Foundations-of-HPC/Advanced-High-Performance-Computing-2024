# Exercise 2: Fix the Vectorization

**Module:** 01 - Compiler Reports
**Difficulty:** Intermediate
**Time:** 20-30 minutes

## Learning Objectives
- Read and interpret compiler vectorization reports
- Identify why vectorization failed
- Apply appropriate fixes
- Verify fixes with compiler reports and benchmarks

## Problem

The following code should vectorize but doesn't. Your task:
1. Compile with vectorization reports
2. Identify why vectorization failed
3. Fix the issue(s)
4. Verify with reports and measurement

```c
// File: broken_code.c
#include <stdio.h>
#include <stdlib.h>

void process_arrays(double *a, double *b, double *result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] * b[i] + 2.0;
    }
}

int main() {
    int n = 10000000;
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *result = malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2.0;
    }
    
    // Call with potential aliasing!
    process_arrays(a, b, result, n);
    process_arrays(a, result, result, n);  
    
    printf("Done\n");
    return 0;
}
```

## Your Tasks

1. **Compile and check reports:**
   ```bash
   gcc -O3 -march=native -fopt-info-vec-all broken_code.c -o broken
   ```

2. **Identify the problem** from compiler output

3. **Fix the code** (multiple possible solutions)

4. **Verify fix:**
   - Compiler reports show vectorization
   - Measure performance improvement

5. **Write a brief explanation** of:
   - What was wrong
   - Why it prevented vectorization
   - How your fix resolved it

## Hints

- Look for "aliasing" in compiler messages
- Consider function signatures
- Think about pointer guarantees

## Solution

See `solutions/exercise2_solution.c` and `solutions/exercise2_explanation.md`
