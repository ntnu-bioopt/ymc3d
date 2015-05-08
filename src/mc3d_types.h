//=======================================================================================================
// Copyright 2015 Asgeir Bjorgan, Matija Milanic, Lise Lyngsnes Randeberg
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================================================

#ifndef MC3D_TYPES_H_DEFINED
#define MC3D_TYPES_H_DEFINED

#define SURVIVAL_CHANCE 0.1
#define W_THRESHOLD 0.0001
#define UZ_THRESH 0.99999
#define NUM_OPTPROPS 4 //number of optical properties used: mua, mus, g and n
#define ZERO_THRESH 1.0e-15
#define NUM_PHOTON_STEPS 45
#define THREADS_PER_BLOCK 384
#define TISSUE_TYPE_AIR 0

#ifdef WIN32
//quickfix for windows
#define M_PI 3.14159
#endif

#include "gpumcml_rng.h"
enum AllocType{ALLOC_GPU, ALLOC_HOST};

typedef struct{
	int num_tissue_types; //number of different tissue types
	float *n; //refraction index
	float *mua; //absorption coeff
	float *mus; //scattering coeff
	float *g; //anistropy factor

	AllocType allocWhere; //whether arrays are allocated on host or GPU
} opticalprops_t;

typedef struct{
	float sample_dx, sample_dy, sample_dz; //spatial discretization
	float length_x, length_y, length_z; //length of VOI
	int num_x, num_y, num_z; //number of voxels
	int *tissue_type; //array over tissue types, corresponding to indices in the optical property arrays above
} geometry_t;

#endif
