#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define I2D(num, c, r) ((r)*(num)+(c))

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out) {
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

  for (int j = 1; j < nj - 1; j++) {
    for (int i = 1; i < ni - 1; i++) {
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i - 1, j);
      ip10 = I2D(ni, i + 1, j);
      i0m1 = I2D(ni, i, j - 1);
      i0p1 = I2D(ni, i, j + 1);

      d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
      d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];

      temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
    }
  }
}

int main() {
  int istep;
  // Number of time steps
  int nstep = 200; 

  // Domain size 
  const int ni = 200;
  const int nj = 100;
  float tfac = 8.418e-5; // thermal diffusivity of silver

  float *temp1, *temp2, *temp_tmp;
  const int size = ni * nj * sizeof(float);

  temp1 = (float*)malloc(size);
  temp2 = (float*)malloc(size);

  // Initialize temperature field with random values
  for (int i = 0; i < ni * nj; ++i)
    temp1[i] = temp2[i] = (float)rand() / (float)(RAND_MAX / 100.0f);

  // Time evolution loop: explicit diffusion update applied nstep times 
  for (istep = 0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1, temp2);
    // Swap pointers: temp2 now holds updated field -> make it the new input
    temp_tmp = temp1;
    temp1 = temp2;
    temp2 = temp_tmp;
  }

  printf("Computation completed successfully.\n");

  free(temp1);
  free(temp2);

  return 0;
}

