#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

__global__ void add_forces()
{
    uint idx = threadIdx.x;
    uint idy = threadIdx.y;

    double2 *vtp = (double2 *) ((char *)v + (idy + spy)*pitch + idx + spx);
    double2 vt = *vtp; // grab velocity target from memory

    idx -= rad;
    idy -= rad;
    double s = 1.f / (1.f + pow(idx, 4) + pow(idy, 4));

    vt.x += s * fx; // add forces to velocity components
    vt.y += s * fy;

    *vtp = vt; // update memory location
}


__global__ void advect_velocity(double2 *v, double *vx, double *vy, int2 domain)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * ()

    if (idx >= domain.x) return;

    double2 vtex, ploc;

    for (uint i = 0; i < tpr; i++) {
        uint j = idy + i;

        if (j >= domain.y) return;

        uint k = j * pad + idx;

        vtex = tex2D<double2>(texObject, (double) idx, (double) j);

        ploc.x = (idx + .5f) - (dt*vtex.x*dx); // bilinear interpolation in velocity space
        ploc.y = (idx + .5f) - (dt*vtex.y*dy);

        vtex = tex2D<double2>(texObject, ploc.x, ploc.y);

        vx[k] = vtex.x;
        vx[k] = vtex.y;
    }
}

__global__ void diffuse_projection()
{
    uint idx = threadIdx.x;
    uint idy = threadIdx.y;
}

__global__ void update_velocity()
{

}

__global__ advect_particles()
{

}
