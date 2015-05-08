#ifndef MC3D_RNG_H_DEFINED
#define MC3D_RNG_H_DEFINED

typedef unsigned long long UINT64;
typedef unsigned int UINT32;

typedef struct{
	int num_rngs;
	UINT64 *rng_x;
	UINT32 *rng_a;
} rng_state_t;

void rng_init(rng_state_t *rng_state, UINT64 seed, int num_rngs);
void rng_free(rng_state_t *rng_state);

#endif
