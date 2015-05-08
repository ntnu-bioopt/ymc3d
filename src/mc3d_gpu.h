#ifndef MC3D_GPU_H_DEFINED
#define MC3D_GPU_H_DEFINED

#include "mc3d_types.h"
#include "mc3d_photons.h"
#include "mc3d_rng.h"

//MAIN TRACKER FUNCTION
void run_3dmc_gpu(geometry_t geometry, opticalprops_t optProps, int num_photons, double *R, double *R_tot, double *T, double *T_tot, float *A, int *num_finished_photons);

//HELPER FUNCTIONS

//reinitialize frozen photons with respect to the given illumination conditions
__global__ void photontrack_reinitialize(geometry_t geometry, photon_properties_t photons, rng_state_t rngseeds, float n0, float n1);

//track photons for a given amount of steps. Break out of for loop when photon is finished if shouldBreak is set. 
__global__ void photontrack_step(geometry_t geometry, opticalprops_t optProps, photon_properties_t photons, rng_state_t rngseeds, int steps, float *A, bool shouldBreak);

//get tissue type at given voxel grid coordinates
__device__ int photontrack_get_tissue_type(int i, int j, int k, float z, float uz);

//assuming that photon continues in direction ux, uy, uz from x, y, z, find which voxel boundary the photon will collide with (is set explicitly to i, j, k and x, y, z) and return the distance to the boundary
__device__ float photontrack_intersection(geometry_t geometry, int *i, int *j, int *k, float *x, const float *ux, float *y, const float *uy, float *z, const float *uz);

//return current grid coordinate along the given coordinate direction
__host__ __device__ int photontrack_get_grid_coord(const float *x, const float *ux, const float *sample_dx);

//calculate specular reflection
__device__ float photontrack_calc_specular_reflection(float n0, float n1);

//do mirroring on x if x falls outside length_x or below 0. Returns true if mirroring was performed. 
__device__ bool photontrack_mirror(float *x, float *ux, const float *length_x);

#endif
