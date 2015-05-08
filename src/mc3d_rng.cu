//=======================================================================================================
// Copyright 2015 Asgeir Bjorgan, Matija Milanic, Lise Lyngsnes Randeberg
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================================================

#include "mc3d_rng.h"
#include "gpumcml_rng.cu"
#include <cstdio>

void rng_init(rng_state_t *rng_state, UINT64 seed, int num_rngs){
	//initialize RNG
	UINT64 *x_rng = new UINT64[num_rngs];
	UINT32 *a_rng = new UINT32[num_rngs];
    	if (init_RNG(x_rng, a_rng, num_rngs, "safeprimes_base32.txt", seed)){
		fprintf(stderr, "Couldn't find safeprimes_base32.txt.\n");
		exit(1);
	}
	
	//allocate cuda arrays
	UINT64 *x_rng_dev;
	UINT32 *a_rng_dev;
	cudaMalloc(&x_rng_dev, num_rngs*sizeof(UINT64));
	cudaMalloc(&a_rng_dev, num_rngs*sizeof(UINT32));

	//copy RNG arrays to gpu arrays
	cudaMemcpy(x_rng_dev, x_rng, num_rngs*sizeof(UINT64), cudaMemcpyHostToDevice);
	cudaMemcpy(a_rng_dev, a_rng, num_rngs*sizeof(UINT32), cudaMemcpyHostToDevice);
	
	rng_state->rng_x = x_rng_dev;
	rng_state->rng_a = a_rng_dev;
	rng_state->num_rngs = num_rngs;

	delete [] x_rng;
	delete [] a_rng;
}

void rng_free(rng_state_t *rng_state){
	cudaFree(rng_state->rng_x);
	cudaFree(rng_state->rng_a);
}
