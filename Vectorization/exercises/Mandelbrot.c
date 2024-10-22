#include <complex>
void MandelbrotCPU(float x1, float y1, float x2, float y2,
                    int width, int height, int maxIters, unsigned short * image)
{
  float dx = (x2-x1)/width, dy = (y2-y1)/height;
  for (int j = 0; j < height; ++j)
    for (int i = 0; i < width; ++i)
      {
        complex<float> c (x1+dx*i, y1+dy*j), z(0,0);
        intcoun t = -1;
        while ((++count < maxIters) && (norm(z) < 4.0))
          z = z*z+c;
        *image++ = count;
      }
}

