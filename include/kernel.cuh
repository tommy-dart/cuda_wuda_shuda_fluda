#ifndef __KERNELS_CUH_
#define __KERNELS_CUH_


#include <vector>
#include <complex>
typedef std::complex<float> fcomp;

void copy_device_f2s_to_comps(float2* vx_dev, float2* vy_dev, float2* vx_host, float2* vy_host, std::vector<fcomp> &vxc, std::vector<fcomp> &vyc);
void copy_comps_to_f2s_device(std::vector<fcomp> vxc, std::vector<fcomp> vyc, float2* vx_dev, float2* vy_dev, float2* vx_host, float2* vy_host);

void advect_velocity_host(float2 *v, float *vx, float *vy);
void diffuse_projection_host(float2 *vx, float2 *vy);
void update_velocity_host(float2* vdev, float *vx, float *vy);
void advect_particles_host(float2* p, float2* v);
void add_forces_host(float2 *v, int px, int py, float fx, float fy);


#endif
