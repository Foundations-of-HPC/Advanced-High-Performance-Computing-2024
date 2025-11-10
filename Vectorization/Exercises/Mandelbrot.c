

/* C++ version using std::complex
#ifdef __cplusplus

#include <complex>
using namespace std;

void MandelbrotCPU ( float x1, float y1, float x2, float y2,
		     int width, int height, int maxIters,
		     unsigned short * image )
{
  float dx = (x2-x1)/width;
  float dy = (y2-y1)/height;
  for (int j = 0; j < height; ++j)
    for (int i = 0; i < width; ++i)
      {
	complex<float> c(x1+dx*i, y1+dy*j), z(0,0);
	int count = -1;
	while ((++count < maxIters) && (norm(z) < 4.0))
        z = z*z+c;
      *image++ = count;
      }
}

#else
*/

// C version using C99 complex.h

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "../headers/timing.h"


void MandelbrotCPU ( float x1, float y1, float x2, float y2,
                     int width, int height, int maxIters,
		     unsigned short * image )
{
  float dx = (x2-x1)/width;
  float dy = (y2-y1)/height;
  for (int j = 0; j < height; ++j)
    for (int i = 0; i < width; ++i)
      {
	float complex c = (x1+dx*i) + (y1+dy*j)*I;
	float complex z = 0.0f + 0.0f*I;
	int count = -1;  // FIXED: was "intcoun t"
	while ((++count < maxIters) && (cabsf(z)*cabsf(z) < 4.0f))
	  z = z*z + c;
	*image++ = (unsigned short)count;
      }
}
//#endif

/* ============================================================================
 * VECTORIZABLE VERSION
 * ============================================================================
 * 
 * The above version is hard to vectorize because:
 * 1. Complex arithmetic is not directly supported by vector instructions
 * 2. Variable iteration count (while loop with data-dependent exit)
 * 
 * Better approach: Separate real and imaginary parts, fixed iteration count
 * ============================================================================
 */

void MandelbrotCPU_Vectorizable ( float x1, float y1, float x2, float y2,
                                  int width, int height, int maxIters, 
                                  unsigned short * image )
{
  float dx = (x2-x1)/width;
  float dy = (y2-y1)/height;
  
  for (int j = 0; j < height; j++)
    {
      float cy = y1 + dy * j;
    
      for (int i = 0; i < width; i++)
	{
	  float cx = x1 + dx * i;
	  float zx = 0.0f, zy = 0.0f;
	  int count = 0;
      
	  // Fixed iteration count - easier to vectorize
	  for (int iter = 0; iter < maxIters; iter++)
	    {
	      float zx2 = zx * zx;
	      float zy2 = zy * zy;
        
	      // Check if we've escaped (magnitude squared > 4)
	      if (zx2 + zy2 > 4.0f)
		{
		  count = iter;
		  break;
		}
        
	      // Complex multiplication: (zx + i*zy)^2 + (cx + i*cy)
	      float new_zx = zx2 - zy2 + cx;
	      float new_zy = 2.0f * zx * zy + cy;
	      
	      zx = new_zx;
	      zy = new_zy;
	    }
	  
	  image[j * width + i] = (unsigned short)count;
	}
    }
}

/* ============================================================================
 * FULLY VECTORIZABLE VERSION (no early exit)
 * ============================================================================
 * 
 * This version removes the early exit, making it fully vectorizable
 * Uses masking to avoid updating already-escaped points
 * ============================================================================
 */

void MandelbrotCPU_FullyVectorizable ( float x1, float y1, float x2, float y2,
                                       int width, int height, int maxIters, 
                                       unsigned short * image )
{
  float dx = (x2-x1)/width;
  float dy = (y2-y1)/height;
  
  for (int j = 0; j < height; j++)
    {
      float cy = y1 + dy * j;
    
     #pragma GCC ivdep
     #pragma GCC unroll 4
      for (int i = 0; i < width; i++)
	{
	  float cx = x1 + dx * i;
	  float zx = 0.0f, zy = 0.0f;
	  int count = maxIters;  // Assume all reach max
      
	  // No early exit - all iterations execute
	  for (int iter = 0; iter < maxIters; iter++)
	    {
	      float zx2 = zx * zx;
	      float zy2 = zy * zy;
	      
	      // Check if escaped - but don't break
	      int escaped = (zx2 + zy2 > 4.0f);
	      
	      // Only update count on first escape
	      if (escaped && (count == maxIters) )
		count = iter;
        
	      // Continue iterating even if escaped (wastes work, but vectorizes)
	      float new_zx = zx2 - zy2 + cx;
	      float new_zy = 2.0f * zx * zy + cy;
        
	      // Only update if not escaped (using conditional assignment)
	      zx = escaped ? zx : new_zx;
	      zy = escaped ? zy : new_zy;
	    }
      
	  image[j * width + i] = (unsigned short)count;
	}
    }
}

/* ============================================================================
 * TEST HARNESS
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
  int mode  = (argc > 1) ? atoi(argv[1]) : 0;
  int width = (argc > 2) ? atoi(argv[2]) : 1024;
  int height = (argc > 3) ? atoi(argv[3]) : 768;
  int maxIters = (argc > 4) ? atoi(argv[4]) : 1200;
  
  // Mandelbrot set bounds (classic view)
  float x1 = -2.5f, x2 = 1.0f;
  float y1 = -1.0f, y2 = 1.0f;
  
  unsigned short *image = (unsigned short*)malloc(width * height * sizeof(unsigned short));
  if (!image) {
    fprintf(stderr, "Allocation failed\n");
    return 1;
  }
  
  printf("Computing Mandelbrot set (%d x %d, %d iterations)\n", width, height, maxIters);
  
  // Time the computation
  double start = PCPU_TIME;
  switch( mode )
    {
    case 0: MandelbrotCPU(x1, y1, x2, y2, width, height, maxIters, image); break;
    case 1: MandelbrotCPU_Vectorizable(x1, y1, x2, y2, width, height, maxIters, image); break;
    case 2: MandelbrotCPU_FullyVectorizable(x1, y1, x2, y2, width, height, maxIters, image); break;
    }
  double elapsed = PCPU_TIME - start;
  printf("Computation time: %.3f seconds\n", elapsed);
  
  // Simple verification - count pixels that escaped
  int escaped_count = 0;
  for (int i = 0; i < width * height; i++) {
    if (image[i] < maxIters) escaped_count++;
  }
  printf("Pixels NOT in the Mandelbrot set: %d / %d (%.1f%%)\n", 
         escaped_count, width*height, 100.0*escaped_count/(width*height));
  
  free(image);
  return 0;
}

/* ============================================================================
 * COMPILE AND TEST:
 * 
 * gcc -O3 -march=native -ftree-vectorize -fopt-info-vec-all \
 *     mandelbrot_fixed.c -o mandelbrot_fixed -lm
 * 
 * ./mandelbrot_fixed 1920 1080 256
 * 
 * EXPECTED:
 * - Original version: May not vectorize (complex numbers, early exit)
 * - Vectorizable version: Better, but still has early exit
 * - Fully vectorizable: Should vectorize well (check compiler reports)
 * 
 * PERFORMANCE NOTE:
 * The fully vectorizable version does MORE work (continues after escape),
 * but may be faster due to vectorization. Profile to verify!
 * ============================================================================
 */
