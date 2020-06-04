#ifndef __KERNELS_CUH_
#define __KERNELS_CUH_

#include "defines.h"

texture<float2, 2> texObj;

__global__ void advect_velocity(double2 *v, double2 *vx, double2 *vy, int domain, int pad, int dt, int tpr);


#endif
