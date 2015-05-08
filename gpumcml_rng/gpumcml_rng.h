#ifndef GPUMCML_RNG_H_DEFINED
#define GPUMCML_RNG_H_DEFINED

#include "mc3d_rng.h"

__device__ float rand_MWC_co(UINT64* x,UINT32* a);
__device__ float rand_MWC_oc(UINT64* x,UINT32* a);
int init_RNG(UINT64 *x, UINT32 *a, const UINT32 n_rng, const char *safeprimes_file, UINT64 xinit);


#endif
