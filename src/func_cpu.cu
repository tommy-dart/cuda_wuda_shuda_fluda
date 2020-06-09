#include "defines.h"

#include <vector>
#include <complex>
#include <math.h>
#include <cuda_runtime.h>

using namespace std;

typedef complex<float> fcomp;

const float PI = acos(-1);


fcomp float2_to_fcomp(float2 f2) {
    fcomp fc(f2.x, f2.y);
    return fc;
}


void copy_f2s_to_comps(float2* vxf, float2* vyf, vector<fcomp> &vxc, vector<fcomp> &vyc) {
    for (int i = 0; i < DIM_FFT_DATA; i++) {
        vxc[i] = float2_to_fcomp(vxf[i]);
        vyc[i] = float2_to_fcomp(vyf[i]);
    }
}


float2 fcomp_to_float2(fcomp fc) {
    float2 f2;
    f2.x = real(fc);
    f2.y = imag(fc);
    return f2;
}


void copy_comps_to_f2s(vector<fcomp> vxc, vector<fcomp> vyc, float2* vxf, float2* vyf) {
    for (int i = 0; i < DIM_FFT_DATA; i++) {
        vxf[i] = fcomp_to_float2(vxc[i]);
        vyf[i] = fcomp_to_float2(vyc[i]);
    }
}


void fft_cpu(vector<fcomp> &x, bool invert) {
    int n = x.size();

    if (n == 1)  // base case for recursion
        return;

    vector<fcomp> x0(n/2), x1(n/2);
    for (int i = 0; i * 2 < n; i++) {
        x0[i] = x[2*i];
        x1[i] = x[2*i+1];
    }

    fft_cpu(x0, invert);
    fft_cpu(x1, invert);

    float ang = 2*PI/n;
    if (invert) ang = -ang;

    fcomp omega(cos(ang), sin(ang));  // wavelength
    fcomp w(1);

    for (int i = 0; i*2 < n; i++) {
        x[i] = x0[i] + w * x1[i];
        x[i+n/2] = x0[i] - w * x1[i];

        if (invert) {  // since this is done at every level of binary recursion, ends up dividing both indices by n
            x[i] /= 2;
            x[i + n/2] /= 2;
        }
        w *= omega;
    }
}


void advect_velocity_cpu(float2* v, float* vx, float* vy, int dim, float dt) {
    int2 pprev;  // previous voxel position
    float2 vprev;  // previous voxel velocity

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            int ind = i * dim + j;

            // trace back to previous position using velocity at current position
            vprev = v[ind];
            pprev.x = int((i + 0.5f) - (dt * vprev.x * dim));
            pprev.y = int((j + 0.5f) - (dt * vprev.y * dim));

            // wrap around the border
            if (pprev.x > dim) pprev.x -= dim;
            if (pprev.y > dim) pprev.y -= dim;
            if (pprev.x < 0) pprev.x += dim;
            if (pprev.y < 0) pprev.y += dim;

            // save velocity from past voxel in component vectors
            int p_ind = pprev.x * dim + pprev.y;
            vprev = v[p_ind];

            vx[ind] = vprev.x;
            vy[ind] = vprev.y;
        }
    }
}


void diffuse_projection_cpu(float2 *vxcomp, float2 *vycomp, int dim, float dt, float visc) {

// complex velocity FFT x- and y-components for computation
    float2 vxc, vyc;  // note the .x and .y attributes of these correspond to real
                      //    and imaginary components for each velocity FFT term

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            int ind = i * dim + j;
            vxc = vxcomp[ind];
            vyc = vycomp[ind];

            // compute index in FFT components
            float iix = (float) i;
            float iiy = (float) j;
            if (j > (dim/2)) iiy -= (float) dim;

            // calculate diffusion constant (diff) based on viscosity with smoothing
            float k2 = iix*iix + iiy*iiy;
            float diff = 1.0f / (1.0f + visc * dt * k2);

            vxc.x *= diff;
            vxc.y *= diff;
            vyc.x *= diff;
            vyc.y *= diff;

            if (k2 > 0.) {
                // if diffusion constant is positive perform velocity projection
                float k2_inv = 1.0f / k2;  // scaling the size of change in frequency domain
                                          // other options on https://www.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft
                float vp_real = (iix*vxc.x + iiy*vyc.x) * k2_inv;
                float vp_imag = (iix*vxc.y + iiy*vyc.y) * k2_inv;

                vxc.x -= vp_real * iix;
                vxc.y -= vp_imag * iix;
                vyc.x -= vp_real * iiy;
                vyc.y -= vp_imag * iiy;
            }

            vxcomp[ind] = vxc;
            vycomp[ind] = vyc;
        }
    }
}


void update_velocity_cpu(float2* v, float* vx, float* vy, int dim) {
    float2 vnew;

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {

            int ind = i * dim + j;

            // scale FFT (other options suggested on mathworks forum https://www.mathworks.com/matlabcentral/answers/15770-scaling-the-fft-and-the-ifft)
            float scale = 1.f;// / ((float)sqrt(dim));
            vnew.x = vx[ind] * scale;
            vnew.y = vy[ind] * scale;

            v[ind] = vnew;
        }
    }
}


void advect_particles_cpu(float2* p, float2* v, int dim, float dt) {
    float2 pt, vt;

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            int ind = i * dim + j;
            pt = p[ind];

            // grab vt at the voxel pt points to
            int pti = (int) (pt.x*dim);
            int ptj = (int) (pt.y*dim);
            int pind = pti * dim + ptj;

            vt = v[pind];

            // update positons
            pt.x += dt * vt.x;
            pt.y += dt * vt.y;

            if (pt.x < 0.f) pt.x += 1.f;
            if (pt.x > 1.f) pt.x -= 1.f;
            if (pt.y < 0.f) pt.y += 1.f;
            if (pt.y > 1.f) pt.y -= 1.f;

            p[ind] = pt;
        }
    }

}


void add_forces_cpu(float2 *v, int dim, int spx, int spy, float fx, float fy, int r) {
    float2 vt;

    for (int i = 0; i < 2*r; i++) {
        for (int j = 0; j < 2*r; j++) {
            int ind = (i + spx) * dim + j + spy;
            vt = v[ind];
            float s = 1.f / (1.f + pow(i-r, 4) + pow(j-r, 4));

            vt.x += s * fx;
            vt.y += s * fy;

            v[ind] = vt;
        }
    }
}
