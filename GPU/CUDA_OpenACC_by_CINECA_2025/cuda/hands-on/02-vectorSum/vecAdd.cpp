#include <stdio.h>
#include <stdlib.h>

void initWith(float num, float *a, int N) {
  for (int i = 0; i < N; ++i)
    a[i] = num;
}

void addVectors(float *result, float *a, float *b, int N) {
  for (int i = 0; i < N; ++i)
    result[i] = a[i] + b[i];
}

void checkElementsAre(float target, float *array, int N) {
  for (int i = 0; i < N; i++) {
    if (array[i] != target) {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main() {
  const int N = 2 << 20; // 2097152 elements
  size_t size = N * sizeof(float);

  float *a, *b, *c;

  // Allocate memory
  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(size);

  // Initialize arrays
  initWith(3.0, a, N);
  initWith(4.0, b, N);

  // Perform addition
  addVectors(c, a, b, N);

  // Verify results
  checkElementsAre(7.0, c, N);

  // Free memory
  free(a);
  free(b);
  free(c);
  
  return 0;
}

