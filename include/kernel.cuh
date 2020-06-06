#ifndef __KERNELS_CUH_
#define __KERNELS_CUH_

// #include "defines.h"

// texture<float2, 2> texObj;

__global__ void advect_velocity(float2 *v, float *vx, float *vy, int domain, int pad, int dt, int tpr);
__global__ void diffuse_projection(float2 *vx, float2 *vy, int domain, int dt, float visc, int tpr);
__global__ void update_velocity(float2* v, float *vx, float *vy, int domain, int pad, int dt, int tpr, size_t pitch);\
__global__ void advect_particles(float2* p, float2* v, int domain, int tp, int tpr, size_t t_pitch);

void update_velocity_cpu(float2* vdev, float *vx, float *vy, int domain, int pad, int dt, int tpr);
void advect_particles_cpu(float2* p, float2* vdev, int domain, int dt, int tpr);


void bindTexture();
void updateTexture(float2 *vdev);
void setupTexture(int x, int y);

#endif
