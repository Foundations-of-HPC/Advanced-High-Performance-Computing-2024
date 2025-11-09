
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "../headers/timing.h"

typedef struct { float x,y,z,w; } Particle;

typedef struct { float *x,*y,*z,*w; } ParticleSoA;

void update_aos(Particle* p, int n, float dt)
{
  for(int i=0;i<n;i++)
    {
      p[i].x += dt*p[i].w;
      p[i].y += dt*p[i].w;
      p[i].z += dt*p[i].w;
    }
}

void update_soa(ParticleSoA p, int n, float dt)
{
 #pragma omp simd
  for(int i=0;i<n;i++){
    p.x[i] += dt*p.w[i];
    p.y[i] += dt*p.w[i];
    p.z[i] += dt*p.w[i];
  }
}

int main ( int argc, char **argv )
{
  int n = (argc>1?atoi(argv[1]):1000000);
  Particle* aos = (Particle*)aligned_alloc(64, n*sizeof(Particle));
  ParticleSoA soa =
    {
      .x=(float*)aligned_alloc(64, n*sizeof(float)),
      .y=(float*)aligned_alloc(64, n*sizeof(float)),
      .z=(float*)aligned_alloc(64, n*sizeof(float)),
      .w=(float*)aligned_alloc(64, n*sizeof(float))
    };
  
  for(int i=0;i<n;i++)
    {
      aos[i].x=aos[i].y=aos[i].z=0; aos[i].w=1.0f;
      soa.x[i]=soa.y[i]=soa.z[i]=0; soa.w[i]=1.0f;
    }

  double elapsed = PCPU_TIME;
  update_aos(aos,n,0.01f);
  elapsed = PCPU_TIME - elapsed;
  printf("aos: %g s\n", elapsed );
  
  elapsed = PCPU_TIME;
  update_soa(soa,n,0.01f);
  elapsed = PCPU_TIME - elapsed;
  printf("soa: %g s\n", elapsed );
  
  free(aos);
  free(soa.x);
  free(soa.y);
  free(soa.z);
  free(soa.w);

  return 0;
}
