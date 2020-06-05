#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include <helper_cuda.h>    // checkCudaErrors

#include "defines.h"


// #include "kernel.cuh"
texture<float2, 2> texObj;
size_t t_pitch;
static cudaArray *texArray = NULL;


__global__ void add_forces()
{
    uint idx = threadIdx.x;
    uint idy = threadIdx.y;

    // float2 *vtp = (float2 *) ((char *)v + (idy + spy)*pitch + idx + spx);
    // float2 vt = *vtp; // grab velocity target from memory

    // idx -= rad;
    // idy -= rad;
    // double s = 1.f / (1.f + pow(idx, 4) + pow(idy, 4));

    // vt.x += s * fx; // add forces to velocity components
    // vt.y += s * fy;

    // *vtp = vt; // update memory location
}


__global__ void advect_velocity(float2 *v, float *vx, float *vy, int domain, int pad, int dt, int tpr)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * (tpr * blockDim.y) + threadIdx.y * tpr;

    if (idx >= domain) return;

    // has to be float2, b/c o conversion from float2 to float2
    float2 vtex, ploc;

    for (uint i = 0; i < tpr; i++) {
        uint j = idy + i;

        if (j >= domain) return;

        uint k = j * pad + idx;

        vtex = tex2D(texObj, (float) idx, (float) j);

        ploc.x = (idx + .5f) - (dt*vtex.x*domain); // bilinear interpolation in velocity space
        ploc.y = (idx + .5f) - (dt*vtex.y*domain);

        vtex = tex2D<float2>(texObj, ploc.x, ploc.y);

        vx[k] = vtex.x;
        vx[k] = vtex.y;
    }
}


__global__ void diffuse_projection(float2 *vx, float2 *vy, int domain, int dt, float visc, int tpr)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * (tpr * blockDim.y) + threadIdx.y * tpr;

    if (idx >= domain) return;

    float2 xterm, yterm;

    for (uint i = 0; i < tpr; i++) {
        uint j = idy + i;

        if (j >= domain) return;

        uint k = j * domain + idx;
        xterm = vx[k];
        yterm = vy[k];

        int jx = idx;
        int jy = 0;

        float k_squared = (float)(jx * jx + jy * jy);
        float diff = 1.0 / (1.0 + visc * dt * k_squared);
        xterm.x *= diff;
        xterm.y *= diff;
        yterm.x *= diff;
        yterm.y *= diff;

        if (k_squared > 0.0)
        {
            float inv_k_square = 1.0 / k_squared;
            float real_k_proj = (jx * xterm.x + jy * yterm.x);
            float imag_k_proj = (jx * xterm.y + jy * yterm.y);

            xterm.x -= real_k_proj * real_k_proj * jx;
            xterm.y -= real_k_proj * imag_k_proj * jx;
            yterm.x -= real_k_proj * real_k_proj * jy;
            yterm.y -= real_k_proj * imag_k_proj * jy;
        }

        vx[k] = xterm;
        vy[k] = yterm;
    }
}



__global__ void update_velocity(float2* v, float *vx, float *vy, int domain, int pad, int dt, int tpr, size_t pitch)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * (tpr * blockDim.y) + threadIdx.y * tpr;

    if (idx >= domain) return;

    float2 nvterm;

    for (uint i = 0; i < tpr; i++) {
        uint j = idy + i;

        if (j >= domain) return;

        uint k = j * pad + idx;

        float scale = 1.0 / (domain * domain);

        nvterm.x = vx[k] * scale;
        nvterm.y = vy[k] * scale;

        float2 *vel = (float2 *)((char *)v + j * pitch) + idx;
        *vel = nvterm;
    }
}


void update_velocity_cpu(float2* vdev, float *vx, float *vy, int domain, int pad, int dt, int tpr)
{
    dim3 grid(TW, TH);
    dim3 block(BLOCKDX, BLOCKDY);

    update_velocity<<<grid, block>>>(vdev, vx, vy, domain, pad, dt, tpr, t_pitch);
}


__global__ void advect_particles(float2* p, float2* v, int domain, int dt, int tpr, size_t t_pitch)
{

}


void advect_particles_cpu(float2* p, float2* vdev, int domain, int dt, int tpr)
{
    dim3 grid(TW, TH);
    dim3 block(BLOCKDX, BLOCKDY);

    advect_particles<<<grid, block>>>(p, vdev, domain, dt, tpr, t_pitch);
}


void bindTexture(void)
{
    cudaBindTextureToArray(texObj, texArray);
    getLastCudaError("cudaBindTexture failed");
}


void updateTexture(float2 *vdev)
{
    // cout << DIM << endl;
    cudaMemcpy2DToArray(texArray, 0, 0, vdev, t_pitch, DIM*sizeof(float2), DIM, cudaMemcpyDeviceToDevice);
    // getLastCudaError("cudaMemcpy failed");
}


void setupTexture(int x, int y)
{
    texObj.filterMode = cudaFilterModeLinear;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float2>();

    cudaMallocArray(&texArray, &desc, y, x);
    getLastCudaError("cudaMalloc failed");

    // cudaResourceDesc texRes;
    // memset(&texRes,0,sizeof(cudaResourceDesc));
    //
    // texRes.resType = cudaResourceTypeArray;
    // texRes.res.array.array = array;
    // //
    //
    // cudaTextureDesc texDescr;
    // memset(&texDescr,0,sizeof(cudaTextureDesc));
    //
    // texDescr.normalizedCoords = false;
    // texDescr.filterMode       = cudaFilterModeLinear;
    // texDescr.addressMode[0] = cudaAddressModeWrap;
    // texDescr.readMode = cudaReadModeElementType;
    //
    // checkCudaErrors(cudaCreateTextureObject(&texObj, &texRes, &texDescr, NULL));
}
