#ifndef MC3D_GPU_H_DEFINED
#define MC3D_GPU_H_DEFINED

#include "mc3d_types.h"

__host__ __device__ float intersection(const float *bmin_x, const float *bmax_x, const float *bmin_y, const float *bmax_y, const float *bmin_z, const float *bmax_z, float *x, const float *dx, float *y, const float *dy, float *z, const float *dz); //find distance to bounding box. x,y,z are updated as the intersection point
__host__ __device__ int getGridCoord(const float *x, const float *ux, const float *sample_dx); //return grid coordinate corresponding to input coordinate
__host__ __device__ float calcRSp(float n0, float n1); //specular reflection

#ifdef FINGER_SIMULATION
void run_3dmc_gpu(Geometry geometry, OpticalProps optProps, int num_photons, double *R, double *R_tot, double *T, double *T_tot, float *A, double *B, int *num_finished_photons, float arclength);
#else
void run_3dmc_gpu(Geometry geometry, OpticalProps optProps, int num_photons, double *R, double *R_tot, double *T, double *T_tot, float *A, double *B, int *num_finished_photons);
#endif

#endif
