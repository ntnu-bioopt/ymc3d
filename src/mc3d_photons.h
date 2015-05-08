#ifndef MC3D_PHOTONS_H_DEFINED
#define MC3D_PHOTONS_H_DEFINED

#include "mc3d_types.h"
struct Photons{
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
};
void initializePhotons(Photons *photons, int num_photons, AllocType allocType);

void deinitializePhotons(Photons *photons);


//update R and T with refl and transm from finished photons, depending on depth
void detector(int photon, Geometry geometry, Photons photons, double *R, double *totR, double *T, double *totT);
void beam(int i, Geometry geometry, Photons photons, double *B);

#define BEAM_DONT_ADD_COORDINATES -1 //for setting the weight to -1 if the photon position in the beam photon thing should be saved

#endif
