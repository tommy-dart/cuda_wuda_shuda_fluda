#ifndef __FUNC_CPU_H_
#define __FUNC_CPU_H_

#include "defines.h"

#include <complex>
#include <vector>
typedef std::complex<float> fcomp;

fcomp float2_to_fcomp(float2 f2);
void copy_f2s_to_comps(float2* vxf, float2* vyf, std::vector<fcomp> &vxc, std::vector<fcomp> &vyx);

float2 fcomp_to_float2(fcomp fc);
void copy_comps_to_f2s(std::vector<fcomp> vxc, std::vector<fcomp> vyc, float2* vxf, float2* vyf);

void fft_cpu(std::vector<fcomp> &x, bool invert);

void advect_velocity_cpu(float2* v, float* vx, float* vy, int dim, float dt);
void diffuse_projection_cpu(float2 *vxcomp, float2 *vycomp, int dim, float dt, float visc);
void update_velocity_cpu(float2* v, float* vx, float* vy, int dim);
void advect_particles_cpu(float2* p, float2* v, int dim, float dt);
void add_forces_cpu(float2 *v, int dim, int spx, int spy, float fx, float fy, int r);

#endif
