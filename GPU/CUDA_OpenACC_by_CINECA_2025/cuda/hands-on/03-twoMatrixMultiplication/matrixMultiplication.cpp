#include <stdio.h>
#include <stdlib.h>

#define N 1024

void matrixMulCPU(int *a, int *b, int *c) {
  int val = 0;
  for (int row = 0; row < N; ++row)
    for (int col = 0; col < N; ++col) {
      val = 0;
      for (int k = 0; k < N; ++k)
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
    }
}

int main() {
  int *a, *b, *c;
  int size = N * N * sizeof(int);

  // Allocate memory
  a = (int *)malloc(size);
  b = (int *)malloc(size);
  c = (int *)malloc(size);

  // Initialize memory
  for (int row = 0; row < N; ++row)
    for (int col = 0; col < N; ++col) {
      a[row * N + col] = row;
      b[row * N + col] = col + 2;
      c[row * N + col] = 0;
    }

  // Perform matrix multiplication
  matrixMulCPU(a, b, c);

  // Print a success message
  printf("Matrix multiplication completed successfully.\n");

  // Free memory
  free(a);
  free(b);
  free(c);
  
  return 0;
}

