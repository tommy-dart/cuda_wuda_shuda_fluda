#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>

#include <helper_cuda.h>    // checkCudaErrors

#include "defines.h"
#include "func_cpu.cuh"
#include <iostream>
#include <complex>
#include <vector>

using namespace std;

extern GLuint vbo;

float2 *vxfield = NULL;
float2 *vyfield = NULL;


void copy_device_f2s_to_comps(float2* vx_dev, float2* vy_dev, float2* vx_host, float2* vy_host, vector<fcomp> &vxc, vector<fcomp> &vyc) {
    cudaMemcpy(vx_host, vx_dev, sizeof(float2)*DIM_FFT_DATA, cudaMemcpyDeviceToHost);
    cudaMemcpy(vy_host, vy_dev, sizeof(float2)*DIM_FFT_DATA, cudaMemcpyDeviceToHost);
    getLastCudaError("cudaMemcpy device_to_host_fft failed");

    for (int i = 0; i < DIM_FFT_DATA; i++) {
        vxc[i] = float2_to_fcomp(vx_host[i]);
        vyc[i] = float2_to_fcomp(vy_host[i]);
    }
}


void copy_comps_to_f2s_device(vector<fcomp> vxc, vector<fcomp> vyc, float2* vx_dev, float2* vy_dev, float2* vx_host, float2* vy_host) {
    for (int i = 0; i < DIM_FFT_DATA; i++) {
        vx_host[i] = fcomp_to_float2(vxc[i]);
        vy_host[i] = fcomp_to_float2(vyc[i]);
    }

    cudaMemcpy(vx_dev, vx_host, sizeof(float2)*DIM_FFT_DATA, cudaMemcpyHostToDevice);
    cudaMemcpy(vy_dev, vy_host, sizeof(float2)*DIM_FFT_DATA, cudaMemcpyHostToDevice);
    getLastCudaError("cudaMemcpy host_to_device_fft failed");
}


__global__ void add_forces(float2 *v, int px, int py, float fx, float fy, int rad, int dim)
{
    uint idx = threadIdx.x;
    uint idy = threadIdx.y;

    uint ind = (idy + py) * dim + idx + px;
    float2 vt = v[ind]; // grab velocity target from memory

    idx -= rad;
    idy -= rad;
    float s = 1.f / (1.f + pow(idx, 4) + pow(idy, 4));

    vt.x += s * fx; // add forces to velocity components
    vt.y += s * fy;

    v[ind] = vt;
}


void add_forces_host(float2 *v, int px, int py, float fx, float fy)
{
    dim3 block(2*FRADIUS+1, 2*FRADIUS+1);

    printf("add_forces_host\n");

    add_forces<<<1, block>>>(v, px, py, fx, fy, FRADIUS, DIM);
    getLastCudaError("add_forces failed.");
}


__global__ void advect_velocity(float2 *v, float *vx, float *vy, int dim, int dt, int tpr)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x; // column
    uint idy = blockIdx.y * (tpr * blockDim.y) + threadIdx.y * tpr;

    if (idx >= dim) return;

    float2 vprev;
    int2 pprev;

    for (uint i = 0; i < tpr; i++) {
        uint j = idy + i; //row

        if (j >= dim) return;

        uint k = j * dim + idx;

        vprev = v[k];

        pprev.x = (int)(idx + 0.5f) - (dt * vprev.x * dim); // bilinear interpolation in velocity space
        pprev.y = (int)(j + 0.5f) - (dt * vprev.y * dim);

        if (pprev.x > dim) pprev.x -= dim;
        if (pprev.y > dim) pprev.y -= dim;
        if (pprev.x < 0) pprev.x += dim;
        if (pprev.y < 0) pprev.y += dim;

        int p_ind = pprev.x * dim + pprev.y;
        vprev = v[p_ind];

        vx[k] = vprev.x;
        vy[k] = vprev.y;
    }
}

void advect_velocity_host(float2 *v, float *vx, float *vy)
{
    dim3 grid((DIM/TW) + ((DIM%TW) ? 1:0), (DIM/TH) + ((DIM%TH) ? 1:0));
    dim3 block(BLOCKDX, BLOCKDY);

    printf("advect_velocity_host\n");

    advect_velocity<<<grid, block>>>(v, vx, vy, DIM, DT, TH/BLOCKDY);
    getLastCudaError("advect_velocity failed");
}


__global__ void diffuse_projection(float2 *vxcomp, float2 *vycomp, int dim, int dt, float visc, int tpr)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * (tpr * blockDim.y) + threadIdx.y * tpr;

    if (idx >= dim) return;

    float2 vxc, vyc;

    for (uint i = 0; i < tpr; i++) {
        uint j = idy + i;

        if (j >= dim) return;

        uint k = j * dim + idx;
        vxc = vxcomp[k];
        vyc = vycomp[k];

        int iix = idx;
        int iiy = j;
        if (j > (dim/2)) iiy -= dim;

        float k2 = (float)(iix * iix + iiy * iiy);
        float diff = 1.0 / (1.0 + visc * dt * k2);

        vxc.x *= diff;
        vxc.y *= diff;
        vyc.x *= diff;
        vyc.y *= diff;

        if (k2 > 0.0f)
        {
            float k2_inv = 1.0f / k2;
            float real_proj = (iix * vxc.x + iiy * vyc.x) * k2_inv;
            float imag_proj = (iix * vxc.y + iiy * vyc.y) * k2_inv;

            vxc.x -= real_proj * iix;
            vxc.y -= imag_proj * iix;
            vyc.x -= real_proj * iiy;
            vyc.y -= imag_proj * iiy;
        }

        vxcomp[k] = vxc;
        vycomp[k] = vyc;
    }
}


void diffuse_projection_host(float2 *vx, float2 *vy)
{
    dim3 grid((DIM/TW) + ((DIM%TW) ? 1:0), (DIM/TH) + ((DIM%TH) ? 1:0));
    dim3 block(BLOCKDX, BLOCKDY);

    printf("diffuse_projection_host\n");

    diffuse_projection<<<grid, block>>>(vx, vy, DIM, DT, VISC, TH/BLOCKDY);
    getLastCudaError("diffuse_projection failed");

}


__global__ void update_velocity(float2* v, float *vx, float *vy, int dim, int dt, int tpr)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * (tpr * blockDim.y) + threadIdx.y * tpr;

    if (idx >= dim) return;

    float2 vnew;

    for (uint i = 0; i < tpr; i++) {
        uint j = idy + i;

        if (j >= dim) return;

        uint k = j * dim + idx;

        float scale = 0.5;  // can be modified to many different values
        // NOTE: 1.f or greater is very unstable

        vnew.x = vx[k] * scale;
        vnew.y = vy[k] * scale;

        v[k] = vnew;
    }
}


void update_velocity_host(float2* v, float *vx, float *vy)
{
    dim3 grid((DIM/TW) + ((DIM%TW) ? 1:0), (DIM/TH) + ((DIM%TH) ? 1:0));
    dim3 block(BLOCKDX, BLOCKDY);

    printf("update_velocity_host\n");

    update_velocity<<<grid, block>>>(v, vx, vy, DIM, DT, TH/BLOCKDY);
    getLastCudaError("update_velocity failed");
}


__global__ void advect_particles(float2* p, float2* v, int dim, float dt, int tpr)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint idy = blockIdx.y * (tpr * blockDim.y) + threadIdx.y * tpr;

    if (idx >= dim) return;

    float2 pt, vprev;

    for (uint i = 0; i < tpr; i++) {
        uint j = idy + i;

        if (j >= dim) return;

        int k = j * dim + idx;
        pt = p[k];

        int xvi = ((int)(pt.x * dim));
        int yvi = ((int)(pt.y * dim));

        int v_ind = yvi*dim + xvi;
        vprev = v[v_ind];

        pt.x += dt * vprev.x;
        pt.y += dt * vprev.y;

        if (pt.x < 0.f) pt.x += 1.f;
        if (pt.x > 1.f) pt.x -= 1.f;
        if (pt.y < 0.f) pt.y += 1.f;
        if (pt.y > 1.f) pt.y -= 1.f;

        p[k] = pt;
    }
}


void advect_particles_host(float2* p, float2* v)
{
    dim3 grid((DIM/TW) + ((DIM%TW) ? 1:0), (DIM/TH) + ((DIM%TH) ? 1:0));
    dim3 block(BLOCKDX, BLOCKDY);

    printf("advect_particles_host\n");

    advect_particles<<<grid, block>>>(p, v, DIM, DT, TH/BLOCKDY);
    getLastCudaError("advect_particles failed");

}
