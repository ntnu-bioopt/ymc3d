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

struct OpticalProps{
	int num_tissue_types;
	float *n;
	float *mua;
	float *mus;
	float *g;

	AllocType allocWhere; 
};



struct RNGSeeds{
	UINT64 *rng_x;
	UINT32 *rng_a;
};


struct Geometry{
	float sample_dx, sample_dy, sample_dz;
	float length_x, length_y, length_z;
	int num_x, num_y, num_z;
	int *tissue_type;
};

#endif
