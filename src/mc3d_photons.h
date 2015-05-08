#ifndef MC3D_PHOTONS_H_DEFINED
#define MC3D_PHOTONS_H_DEFINED

#include "mc3d_types.h"
typedef struct{
	int num_photons;

	//current photon coordinates
	float *x;
	float *y;
	float *z;

	//direction cosines
	float *ux;
	float *uy;
	float *uz;

	float *W; //photon weights
	float *s; //current photon path lengths
	int *initialized_photons; //number of times a new photon has been initialized
	bool *finished_photons; //array over whether the current photon has either reached ambient medium or been absorbed

	float *absorbed; //absorbed weights

	int *num_finished_photons;

	float *dummy; 

	AllocType allocWhere; //where the arrays are allocated
} photon_properties_t;

void photon_properties_initialize(photon_properties_t *photons, int num_photons, AllocType allocType);

void photon_properties_deinitialize(photon_properties_t *photons);


//update R and T with refl and transm from finished photons, depending on depth
void photon_detector(int photon_ind, geometry_t geometry, photon_properties_t photons, double *R, double *totR, double *T, double *totT);

#endif
