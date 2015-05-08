#include "gpumcml_rng.h"

typedef struct{
	int num_rngs;
	UINT64 *rng_x;
	UINT32 *rng_a;
} rng_state_t;

void rng_init(rng_state_t *rng_state, UINT64 seed, int num_rngs);
void rng_free(rng_state_t *rng_state);
