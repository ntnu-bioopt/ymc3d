#include <cmath>
#include "gpumcml_rng.cu"

#ifndef WIN32
#include <sys/time.h>
#endif

#include <iostream>
#include "mc3d_types.h"
#include "mc3d_io.h"
#include "mc3d_photons.h"
#include <cstdio>
#include <signal.h>
using namespace std;


#ifdef FINGER_SIMULATION
#warning "Compiling for circular irradiation of finger."
#endif

texture<int, 3, cudaReadModeElementType> tissue_type_tex; //3D texture containing spatially resolved tissue types
__device__ int getTissueType(int i, int j, int k, float z, float uz){
	int tissue_type = tex3D(tissue_type_tex, i, j, k);
	#ifdef FINGER_SIMULATION
	return tissue_type;
	#else
	return ((z <= 0) && (uz < 0)) ? TISSUE_TYPE_AIR : tissue_type;
	#endif
}

//from ray tracing on gpus, feist and wende, 2012
//http://www.inf.fu-berlin.de/lehre/SS12/SP-Par/download/FeistWende.pdf
//returns the parameter t in r = r0 + dr*t for intersection with the bounding box (bmin_x, bmin_y, bmin_z)->(bmax_x, bmax_y, bmax_z)
__host__ __device__ float intersection(const float *bmin_x, const float *bmax_x, const float *bmin_y, const float *bmax_y, const float *bmin_z, const float *bmax_z, float *x, const float *dx, float *y, const float *dy, float *z, const float *dz){
	//x-plane intersections
	float t1x = (*bmin_x - *x)/(*dx);
	float t2x = (*bmax_x - *x)/(*dx);
	float tnear = fminf(t1x, t2x);
	float tfar = fmaxf(t1x, t2x);


	//y-plane intersections
	float t1y = (*bmin_y - *y)/(*dy);
	float t2y = (*bmax_y - *y)/(*dy);
	tnear = fmaxf(tnear, fminf(t1y, t2y));
	tfar = fminf(tfar, fmaxf(t1y, t2y));


	//z-plane intersections
	float t1z = (*bmin_z - *z)/(*dz);
	float t2z = (*bmax_z - *z)/(*dz);
	tnear = fmaxf(tnear, fminf(t1z, t2z));
	tfar = fminf(tfar, fmaxf(t1z, t2z));
	
	
	//the intersection t for the photon must be the largest one (> 0)
	float t = fmaxf(tnear, tfar);
	
	//avoid catastrophic numerical inaccuracies by explicitly setting each coordinate to the boundary values
	*x = ((t == t1x) || (t == t2x)) ? ((t == t1x) ? *bmin_x : *bmax_x) : *x + *dx*t;
	*y = ((t == t1y) || (t == t2y)) ? ((t == t1y) ? *bmin_y : *bmax_y) : *y + *dy*t;
	*z = ((t == t1z) || (t == t2z)) ? ((t == t1z) ? *bmin_z : *bmax_z) : *z + *dz*t;
	return t;
}

__device__ float intersection(Geometry geometry, int *i, int *j, int *k, float *x, const float *ux, float *y, const float *uy, float *z, const float *uz){
	//x-plane intersections
	float border_x = ((*ux >= 0) ? (*i+1) : *i)*geometry.sample_dx;
	float t_x = (border_x - *x)/(*ux);

	//y-plane intersections 
	float border_y = ((*uy >= 0) ? (*j+1) : *j)*geometry.sample_dy;
	float t_y = (border_y - *y)/(*uy);

	//z-plane intersections
	float border_z = ((*uz >= 0) ? (*k+1) : *k)*geometry.sample_dz;
	float t_z = (border_z - *z)/(*uz);

	float t = fminf(t_x, fminf(t_y, t_z));

	//avoid numerical inaccuracies
	*x = (t == t_x) ? border_x : *x + *ux*t;
	*y = (t == t_y) ? border_y : *y + *uy*t;
	*z = (t == t_z) ? border_z : *z + *uz*t;

	return t;
}


//sample_dx is grid step
__host__ __device__ int getGridCoord(const float *x, const float *ux, const float *sample_dx){
	int i = floorf(*x/(*sample_dx)); //initial estimate. Assumes that the coordinate is either within bounding box or on the boundary with positive direction. Rounds down
	i += (*x == (i+1)*(*sample_dx)); //quickfix numerical errors
	float bmin = i*(*sample_dx);

	//check whether position is on the bmin boundaries (will never be on the bmax boundaries since that would be the bmin of the next box
	//if it does, determine whether indices should stay the same or decremented according to the direction
	i = ((*x > bmin) || ((*x == bmin) && (*ux >= 0))) ? i : i-1;

	return i;
}

//illumination on voxelized finger from circular IACOBUS-thing
//assumes that finger boundaries can be defined by the formula used in the function. Otherwise, will be wrong.
//x0, z0 defines center of finger
//r0 defines radius of finger
//arclength_divPI is the length of the light arc in radians divided by pi
/*
__host__ __device__ void illumination_arc_finger(float arclength_divPI, float x0, float z0, float r0, float *x, float *ux, float *y, float *uy, float *z, float *uz, float *n0, float *n1, float *W, float *rnd_x, float *rnd_a){
	//sample random theta, estimate direction cosines
	float theta_divpi = arclength_divPI*rand_MWC_co(&x, &a);
	float sintheta, costheta;
	sincospif(theta_divpi, &sintheta, &costheta);
	*ux = costheta;
	*uz = -sintheta;
	*uy = 0;

	//if we assume a perfectly symmetric finger, we can assume that there should be normal incidence. However,
	//due to voxelization, this is not really correct, but we are going to still assume it by assuming
	//the same, initial direction cosines and move the photons to the correct, corresponding voxel.
	*W = 1-calcRSp(n0, n1); //would be fresnel-tjafs if we do it the voxel way
	*x = 0;

}*/



//illumination from circle IACOBUS thing to arbitrary object: will track photon until it reaches skin and will be slow...
//x0, z0, r0 specifies dimensions of arc
//arclength_divPI is the length of the arc divided by pi
__device__ void illumination_arc_arbitrary(Geometry geometry, float arclength_divPI, float x0, float y0, float z0, float r0, float *x, float *ux, float *y, float *uy, float *z, float *uz, UINT64 *rnd_x, UINT32 *rnd_a){
	//sample random theta, estimate direction cosines

	#if 1
	float theta_divpi = arclength_divPI*rand_MWC_co(rnd_x, rnd_a);
	float sintheta, costheta;
	sincospif(theta_divpi, &sintheta, &costheta);
	*ux = -costheta;
	*uz = -sintheta;
	*uy = 0;

	//get start coordinates, assume distance r0 from center
	*x = r0*costheta + x0;
	*z = r0*sintheta + z0;
	*y = y0;
	#else
	*ux = 0;
	*uy = 0;
	*uz = 1;

	float length = geometry.length_x/4;
	*x = x0;//length*rand_MWC_co(rnd_x, rnd_a) + geometry.length_x/2 - length/2;
	*y = y0;
	*z = 0;
	#endif

	//track photon until it reaches tissue
	int i = getGridCoord(x, ux, &geometry.sample_dx);
	int j = getGridCoord(y, uy, &geometry.sample_dy);
	int k = getGridCoord(z, uz, &geometry.sample_dz);
	int tissue = getTissueType(i, j, k, *z, *uz);

	//FIXME: bresenham's line rasterizing algorithm would be far more efficient than this, but it doesn't matter unless we will need to do simulations like these in the future. 
	while (tissue == TISSUE_TYPE_AIR){
		float bmin_x = i*geometry.sample_dx;
		float bmin_z = k*geometry.sample_dz;
		float bmax_x = (i+1)*geometry.sample_dx;
		float bmax_z = (k+1)*geometry.sample_dz;

		//move photon to bounding box
		float t1x = (bmin_x - *x)/(*ux);
		float t2x = (bmax_x - *x)/(*ux);
		float tnear = fminf(t1x, t2x);
		float tfar = fmaxf(t1x, t2x);

		//z-plane intersections
		float t1z = (bmin_z - *z)/(*uz);
		float t2z = (bmax_z - *z)/(*uz);
		tnear = fmaxf(tnear, fminf(t1z, t2z));
		tfar = fminf(tfar, fmaxf(t1z, t2z));
		
		
		//the intersection t for the photon must be the largest one (> 0)
		float t = fmaxf(tnear, tfar);
		
		//avoid catastrophic numerical inaccuracies by explicitly setting each coordinate to the boundary values
		*x = ((t == t1x) || (t == t2x)) ? ((t == t1x) ? bmin_x : bmax_x) : *x + *ux*t;
		*z = ((t == t1z) || (t == t2z)) ? ((t == t1z) ? bmin_z : bmax_z) : *z + *uz*t;

		//get new tissue type
		i = getGridCoord(x, ux, &geometry.sample_dx);
		k = getGridCoord(z, uz, &geometry.sample_dz);
		tissue = getTissueType(i, j, k, *z, *uz);
	}
}




//calculate specular reflection
__host__ __device__ float calcRSp(float n0, float n1){
	float Rsp = (n0 - n1)/(n0 + n1);
	return Rsp*Rsp;
}


//n0 is refraction index of ambient medium, n1 is refraction index of upper voxels. Assume they won't vary spatially for simplicity
#ifdef FINGER_SIMULATION
__global__ void photon_reinitialize(Geometry geometry, Photons photons, Photons beamPhotons, RNGSeeds rngseeds, float n0, float n1, float x0, float y0, float z0, float r0, float arclength){
#else
__global__ void photon_reinitialize(Geometry geometry, Photons photons, Photons beamPhotons, RNGSeeds rngseeds, float n0, float n1){
#endif
//FIXME: if n1 varies spatially, there will be problems, kis. But it probably won't ever.
//in that case, OpticalProps must be an argument, the n array must be copied to shared memory array,
//the tissue type must be looked up, n must be read according to tissue type...

	//manipulate memory access patterns: reflectance first written to shared array, then written to global memory in orderly fashion
//	extern __shared__ float sh_arr[]; 
//
//	//initialize shared mem arr for reflectance
//	float *sh_refl = sh_arr;
//	int ind = blockDim.x*threadIdx.y + threadIdx.x;
//	if (ind < geometry.num_x*geometry.num_y){
//		sh_refl[ind] = 0;	
//	}
//	
//	//if the grid is larger than block dimension, be sure to initialize also the uninitialized reflectance positions
//	if (blockDim.x*blockDim.y < geometry.num_x*geometry.num_y){
//		ind = blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;
//		if (ind < geometry.num_x*geometry.num_y){
//			sh_refl[ind] = 0;
//		}
//	}
//
//	__syncthreads();

	int ind = (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x + blockDim.x*threadIdx.y + threadIdx.x;
	float newx = 0;
	float newy = 0;
	float newz = 0;
	float Rsp = 2;




	if (photons.finished_photons[ind]){
		//initialize new photon if the former has frozen
		UINT64 x = rngseeds.rng_x[ind];
		UINT32 a = rngseeds.rng_a[ind];

		//save reflectance
		//FIXME: assumes plane object. 
		//FIXME: transmission
//		int i = getGridCoord(photons.x[ind], photons.ux[ind], geometry.sample_dx);
//		int j = getGridCoord(photons.y[ind], photons.uy[ind], geometry.sample_dy);
//		sh_refl[j*geometry.num_y + i] += photons.W[ind];
	
		//reinitialize

		
		#if 1
		//uniform illumination
		//float internal_length_x = geometry.length_x/gridDim.x;
		//float internal_length_y = geometry.length_y/gridDim.y;
		//newx = internal_length_x*(blockIdx.x + rand_MWC_co(&x, &a));
		//newy = internal_length_y*(blockIdx.y + rand_MWC_co(&x, &a));
		
		newx = geometry.length_x*rand_MWC_co(&x, &a);
		newy = geometry.length_y*rand_MWC_co(&x, &a);

		//float diff = geometry.length_x/5;
		//newy = geometry.length_x/2 + rand_MWC_co(&x, &a)*diff - diff;

		#else
		//beam
		float newx = geometry.length_x/2;
		float newy = geometry.length_y/2;
		#endif
		
		//arc illumination
		float newux = 0, newuy = 0, newuz = 1;

		#ifdef FINGER_SIMULATION
		illumination_arc_arbitrary(geometry, arclength, x0, y0, z0, r0, &newx, &newux, &newy, &newuy, &newz, &newuz, &x, &a);
		#endif


		photons.x[ind] = newx, photons.y[ind] = newy, photons.z[ind] = newz; //uniform illumination

		

		Rsp = calcRSp(n0, n1);
		photons.ux[ind] = newux, photons.uy[ind] = newuy, photons.uz[ind] = newuz;
		photons.W[ind] = 1.0f - Rsp;
		photons.s[ind] = -log(rand_MWC_oc(&x, &a));
		(photons.initialized_photons[ind])++;

		rngseeds.rng_x[ind] = x;
		rngseeds.rng_a[ind] = a;
		
		photons.finished_photons[ind] = false;
		
	} 
		
	//for saving the beam
	beamPhotons.x[ind] = newx, beamPhotons.y[ind] = newy;
	beamPhotons.W[ind] = (Rsp > 1) ? BEAM_DONT_ADD_COORDINATES : 1.0f; //this will be -1 if the photon is not reinitialized

	//write shared memory reflectance back to global memory reflectance
}			


__device__ bool mirror(float *x, float *ux, const float *length_x){
	bool isReflected = (*x >= *length_x) || (*x <= 0);
	
	//x > length
	*x = (*x >= *length_x) ? 2*(*length_x) - *x : *x;


	//x < 0
	*x = (*x <= 0) ? fabsf(*x) : *x;

	*ux = (isReflected) ? -1*(*ux) : *ux;

	return isReflected;
}

//#define GPU_CALCREFL
  
#ifdef GPU_CALCREFL
__global__ void photon_step(Geometry geometry, OpticalProps optProps, Photons photons, RNGSeeds rngseeds, int steps, float *A, bool shouldBreak, float *R){
#else
__global__ void photon_step(Geometry geometry, OpticalProps optProps, Photons photons, RNGSeeds rngseeds, int steps, float *A, bool shouldBreak){
#endif
	//shared memory array containing optical properties
	extern __shared__ float sh_optProps[]; 
	float *sh_mua = sh_optProps;
	float *sh_mus = sh_optProps + optProps.num_tissue_types;
	float *sh_g = sh_optProps + 2*optProps.num_tissue_types;
	float *sh_n = sh_optProps + 3*optProps.num_tissue_types;
	if (threadIdx.x == 0){
		for (int i=0; i < optProps.num_tissue_types; i++){
			sh_mua[i] = optProps.mua[i];
			sh_mus[i] = optProps.mus[i];
			sh_g[i] = optProps.g[i];
			sh_n[i] = optProps.n[i];
		}
	}
	__syncthreads();



	int ind = (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x + blockDim.x*threadIdx.y + threadIdx.x;

	//read from arrays
	float x = photons.x[ind];
	float y = photons.y[ind];
	float z = photons.z[ind];
	
	float ux = photons.ux[ind];
	float uy = photons.uy[ind];
	float uz = photons.uz[ind];

	float s = photons.s[ind];
	float W = photons.W[ind];

	UINT32 rnd_a = rngseeds.rng_a[ind];
	UINT64 rnd_x = rngseeds.rng_x[ind];
		
	//start grid coordinates
	bool freeze = photons.finished_photons[ind];
	bool wasFrozen = freeze;
		
	int i = getGridCoord(&x, &ux, &geometry.sample_dx);
	int j = getGridCoord(&y, &uy, &geometry.sample_dy);
	int k = getGridCoord(&z, &uz, &geometry.sample_dz);
	int tissue_type = getTissueType(i, j, k, z, uz); //current tissue type

	float db;

	for (int step = 0; step < steps; step++){
		float newx, newy, newz;
		if (shouldBreak && freeze){
			break;
		}
		
		float ni = sh_n[tissue_type];
		float nt = ni;
		float epsilon;

		//FIXME: mu_a and mu_t read from the current cell (not next cell)
		float mu_a = sh_mua[tissue_type];
		float mu_s = sh_mus[tissue_type];
		float g = sh_g[tissue_type];
		float mu_t = mu_a + mu_s;

		//bounding box coordinates
		/*
		float bmin_x = i*geometry.sample_dx;
		float bmin_y = j*geometry.sample_dy;
		float bmin_z = k*geometry.sample_dz;
		float bmax_x = (i+1)*geometry.sample_dx;
		float bmax_y = (j+1)*geometry.sample_dy;
		float bmax_z = (k+1)*geometry.sample_dz;*/

		//estimate distance to boundary of the box in particle direction
		newx = x;
		newy = y;
		newz = z;
		db = intersection(geometry, &i, &j, &k, &newx, &ux, &newy, &uy, &newz, &uz); //provided ux,uy,uz normalized: db is distance to boundary
		//float db = intersection(&bmin_x, &bmax_x, &bmin_y, &bmax_y, &bmin_z, &bmax_z, &newx, &ux, &newy, &uy, &newz, &uz); //provided ux,uy,uz normalized: db is distance to boundary
		//db = (db == 0) ? 2*s/mu_t : db; //numerical inaccuracies or ambiguous grid coordinates can cause db to be zero, will freeze forever. Just set it to twice the stepsize in that case... 

		//find distance to move
		bool hit_boundary = db*mu_t <= s;
		float ds = hit_boundary ? db*mu_t : s; //travel distance s if boundary is further away, otherwise travel distance to boundary

		//calculate new coordinates
		newx = (hit_boundary) ? newx : x + ux*ds/mu_t;
		newy = (hit_boundary) ? newy : y + uy*ds/mu_t;
		newz = (hit_boundary) ? newz : z + uz*ds/mu_t;

		
		//update s
		s = s - ds;

		
		int nexti = getGridCoord(&newx, &ux, &geometry.sample_dx);
		int nextj = getGridCoord(&newy, &uy, &geometry.sample_dy);
		int nextk = getGridCoord(&newz, &uz, &geometry.sample_dz);
		

		int next_tissue_type = getTissueType(nexti, nextj, nextk, newz, uz);
		nt = sh_n[next_tissue_type];

		
		float scatt_ux, scatt_uy, scatt_uz;
		float transrefl_ux = ux, transrefl_uy = uy, transrefl_uz = uz;

		bool isReflected = false;
		//TRANSMIT/REFLECT
		
		//calculate incident angle on the assumption that photon has passed into the next cell
		
		//for the most of the threads, ni is going to be equal to nt, so we might reduce thread divergence by checking for that.
		if (ni == nt){
			//nothing to see here, move along
			transrefl_ux = (ni == nt) ? ux : transrefl_ux;
			transrefl_uy = (ni == nt) ? uy : transrefl_uy;
			transrefl_uz = (ni == nt) ? uz : transrefl_uz;
		} else { //if a single thread does not have ni != nt, however, all threads will run the next code segment and waste time on nothing, but in case we are lucky it will help. 
			float cos_ai = (i != nexti) ? ux : ((j != nextj) ? uy : uz);// (i != nexti)*ux + ((i == nexti) && (j != nextj))*uy + ((i == nexti) && (j == nextj) && (k != nextk))*uz; //find primary direction based on grid switch. FIXME: Can this become zero? 
			cos_ai = fabsf(cos_ai);
			cos_ai = (cos_ai > 1) ? 1 : cos_ai;

			float ai = acosf(cos_ai); //calculate angle
			float ni_nt = ni/nt;
			
			//calculate reflection coefficient, regardless of having or not passed a refraction index boundary
			float sin_ai = sinf(ai);
			float sin_at = ni_nt*sin_ai;
			float cos_at = sqrtf(1.0f - sin_at*sin_at);

			//float R = getFresnel(sin_ai, sin_at, cos_at, ni, nt);
			float R1 = (sin_ai*cos_at - cos_ai*sin_at)/(sin_ai*cos_at + cos_ai*sin_at);
			//float R1 = (cos_at - ni_nt*cos_ai)/(cos_at + ni_nt*cos_ai);
			float R2 = (cos_ai*cos_at - sin_ai*sin_at)/(cos_ai*cos_at + sin_at*sin_at);
			float R = 0.5*R1*R1*(1.0f + R2*R2);

			R = (sin_at > 1) ? 1 : R; //critical angle
			R = (ni == nt) ? 0 : R;


			epsilon = rand_MWC_co(&rnd_x, &rnd_a);
			isReflected = epsilon < R;

			//decide new direction based on assumption of reflection/transmission, and which cell the photon passed through

			//reflection: negative sign if reflected in primary direction, otherwise the same
			//transmission: u*ni/nt if not primary direction, signum(u)*cosf(at) if primary direction
			//NB: preference ordering! If ux was chosen as primary direction, uy and uz should not be treated as primary angles. They are changed only if the previous wasn't changed..
			transrefl_ux = (isReflected) ? ux*((i == nexti) ? 1 : -1) : ((i != nexti) ? copysignf(cos_at, ux*cos_at) : ux*ni_nt);
			transrefl_uy = (isReflected) ? uy*((j == nextj) ? 1 : -1) : (((j != nextj) && (i == nexti)) ? copysignf(cos_at, uy*cos_at) : uy*ni_nt); 
			transrefl_uz = (isReflected) ? uz*((k == nextk) ? 1 : -1) : (((k != nextk) && (j == nextj) && (i == nexti)) ? copysignf(cos_at, uz*cos_at) : uz*ni_nt);
		
		}

		
		//ABSORPTION

		//absorption calculation
		float dW = (mu_t > 0) ? mu_a/mu_t*W : 0; //will be zero if we still are in an ambient medium
		W = W - dW*(!hit_boundary); //do an absorption if we ain't hittin' any boundaries, and aren't in an ambient medium

		#if 0
		if ((!hit_boundary)){
			int abs_i = ((i >= 0) && (i < geometry.num_x)) ? i : ((i < 0) ? 0 : geometry.num_x - 1);
			int abs_j = ((j >= 0) && (j < geometry.num_y)) ? j : ((j < 0) ? 0 : geometry.num_y - 1);
			int abs_k = ((k >= 0) && (k < geometry.num_z)) ? k : ((k < 0) ? 0 : geometry.num_z - 1);
			atomicAdd(&(A[abs_k*geometry.num_y*geometry.num_x + abs_j*geometry.num_x + abs_i]), dW);
		}
		#endif

		//russian roulette
		epsilon = rand_MWC_co(&rnd_x, &rnd_a);
		float newW = (epsilon < SURVIVAL_CHANCE) ? W/SURVIVAL_CHANCE : 0;
		W = ((W < W_THRESHOLD) && (!hit_boundary) && (mu_t != 0)) ? newW : W;

		//SCATTERING

		//decide new direction based on assumption of scattering
		float costheta = (1-g*g)/(1-g+2*g*rand_MWC_oc(&rnd_x, &rnd_a));
		costheta = 1.0f/(2*g)*(1 + g*g - costheta*costheta);
		float sintheta = sqrtf(1-costheta*costheta);
		float phi_divpi = 2*rand_MWC_co(&rnd_x, &rnd_a); //phi divided by PI
		//phi_divpi *= M_PI;

		float cosphi, sinphi;
		sincospif(phi_divpi, &sinphi, &cosphi);
		float invsqrt = rsqrtf(1-uz*uz);

		//avoid division by small number when uz > 0.99999
		scatt_ux = (fabsf(uz) <= UZ_THRESH) ? sintheta*(ux*uz*cosphi - uy*sinphi)*invsqrt + ux*costheta : sintheta*cosphi;
		scatt_uy = (fabsf(uz) <= UZ_THRESH) ? sintheta*(uy*uz*cosphi + ux*sinphi)*invsqrt + uy*costheta : sintheta*sinphi;
		scatt_uz = (fabsf(uz) <= UZ_THRESH) ? -1*sqrtf(1-uz*uz)*sintheta*cosphi + uz*costheta : copysignf(costheta, costheta*uz);


		//new direction based on actual events
		float newux = (!hit_boundary) ? scatt_ux : transrefl_ux;
		float newuy = (!hit_boundary) ? scatt_uy : transrefl_uy;
		float newuz = (!hit_boundary) ? scatt_uz : transrefl_uz;
	

		//rectify numerical errors
		float zero_thresh = 1 - UZ_THRESH;
		newux = (fabs(newux) < zero_thresh) ? 0 : newux; 
		newuy = (fabs(newuy) < zero_thresh) ? 0 : newuy; 
		newuz = (fabs(newuz) < zero_thresh) ? 0 : newuz; 
		//correct for precision errors when coordinates are close to zero: depending on the situation, will get closer and closer to zero but never truly converge
		/*
		newx = (fabs(newx) < ZERO_THRESH) ? 0 : newx;
		newy = (fabs(newy) < ZERO_THRESH) ? 0 : newy;
		newz = (fabs(newz) < ZERO_THRESH) ? 0 : newz;*/
		
		float newu_norm = rsqrtf(newux*newux + newuy*newuy + newuz*newuz);
		newux *= newu_norm;
		newuy *= newu_norm;
		newuz *= newu_norm;

		//actual new direction based on dead or not and ambient medium (i.e. keep old direction forever if photon is dead or has escaped)
		freeze = ((W <= 0) || (mu_t == 0)) || freeze;
		ux = (!freeze) ? newux : ux;
		uy = (!freeze) ? newuy : uy;
		uz = (!freeze) ? newuz : uz;


		//set new s if necessary
		epsilon = rand_MWC_oc(&rnd_x, &rnd_a);
		float newS = (W > 0) ? -log(epsilon) : 0;
		s = (freeze || hit_boundary) ? s : newS;

		//set new final coordinates
		x = (!freeze) ? newx : x;
		y = (!freeze) ? newy : y;
		z = (!freeze) ? newz : z;
		
		//mirror boundary conditions
		#if 1
		isReflected = isReflected || mirror(&x, &ux, &geometry.length_x) || mirror(&y, &uy, &geometry.length_y);
		#endif
	
		//update grid coordinates
		i = (!((hit_boundary && isReflected) || freeze)) ? nexti : i;
		j = (!((hit_boundary && isReflected) || freeze)) ? nextj : j;
		k = (!((hit_boundary && isReflected) || freeze)) ? nextk : k;

	
		tissue_type = (!((hit_boundary && isReflected) || freeze)) ? next_tissue_type : tissue_type;
	}
	
	if (isinf(x) || isnan(x) || isinf(y) || isnan(y) || isinf(z) || isnan(z) || isinf(s) || isnan(s) || isinf(ux) || isnan(ux) || isinf(uy) || isnan(uy) || isinf(uz) || isnan(uz) || (db == 0)){
		freeze = true;
		x = 0;
		y = 0;
		z = 0;
		ux = 0;
		uy = 0;
		uz = 1;
		W = 0;
		s = 0;
	}




	if ((freeze != wasFrozen) && (freeze)){
		photons.num_finished_photons[ind]++;

		#ifdef GPU_CALCREFL
		if ((z <= 0) && (i >= 0) && (i < geometry.num_x) && (j >= 0) && (j < geometry.num_y)){
			atomicAdd(&(R[j*geometry.num_x + i]), W);
		}
		#endif		
	}
	
	//write back to arrays
	photons.x[ind] = x;
	photons.y[ind] = y;
	photons.z[ind] = z;
	
	photons.ux[ind] = ux;
	photons.uy[ind] = uy;
	photons.uz[ind] = uz;

	photons.s[ind] = s;
	photons.W[ind] = W;

	photons.finished_photons[ind] = freeze;
	
	rngseeds.rng_a[ind] = rnd_a;
	rngseeds.rng_x[ind] = rnd_x;
}

//#include "mc3d_test.cu"


#ifndef WIN32
bool kill_process = false;
void sig_handler(int signo){
	if (signo == SIGINT){
		kill_process = true;
	}
}

#endif

#ifdef FINGER_SIMULATION
void run_3dmc_gpu(Geometry geometry, OpticalProps optProps, int num_photons, double *R, double *R_tot, double *T, double *T_tot, float *A, double *B, int *num_finished_photons, float arclength){
#else
void run_3dmc_gpu(Geometry geometry, OpticalProps optProps, int num_photons, double *R, double *R_tot, double *T, double *T_tot, float *A, double *B, int *num_finished_photons){
#endif
	#ifndef WIN32
	//cath SIGINT signals: sets kill_process to true, which is checked in the monte carlo loop. 
	if (signal(SIGINT, sig_handler) == SIG_ERR){
		printf("\ncan't catch SIGINT\n");
	} 

	#endif


	cudaArray *tissue_type_arr;
	//create 3d texture for tissue types
	cudaExtent extent = make_cudaExtent(geometry.num_x, geometry.num_y, geometry.num_z); //since a cuda array is allocated and participating in a copy, the first dimension is defined in terms of elements (and not bytes, as is defined in the cudaExtent documentation entry...)

	//allocate 3D array
	cudaMalloc3DArray(&tissue_type_arr, &(tissue_type_tex.channelDesc), extent);

	//copy host allocated tissue type array to 3d array
	cudaMemcpy3DParms copyParams = {0};
	copyParams.srcPtr   = make_cudaPitchedPtr((void*)(geometry.tissue_type), geometry.num_x*sizeof(int), geometry.num_x, geometry.num_y);
	copyParams.dstArray = tissue_type_arr;
	copyParams.extent   = extent;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&copyParams);

	//set texture parameters, boundary conditions
	tissue_type_tex.normalized = false; //integer access
	tissue_type_tex.filterMode = cudaFilterModePoint; //no linear interpolation
	tissue_type_tex.addressMode[0] = cudaAddressModeClamp; //clamp texture coordinates: tissue type outside volume of interest is the border tissue type
	tissue_type_tex.addressMode[1] = cudaAddressModeClamp;
	tissue_type_tex.addressMode[2] = cudaAddressModeClamp;

	//bind array to 3D texture
	cudaBindTextureToArray(tissue_type_tex, tissue_type_arr, tissue_type_tex.channelDesc);
	int tissue_type_epidermis = 1;

	#ifdef FINGER_SIMULATION

	//find r0, x0, z0 (estimate of radius of finger, center of finger)
	float y0 = geometry.length_y/2;
	float x0, z0, r0;

	//find center of mass
	float cog_x = 0, cog_z = 0, cog_tot = 0;
	float uy0 = 0;
	int j0 = getGridCoord(&y0, &uy0, &geometry.sample_dy);
	for (int i=0; i < geometry.num_x; i++){
		for (int k=0; k < geometry.num_z; k++){
			int tisstype = geometry.tissue_type[k*geometry.num_y*geometry.num_x + j0*geometry.num_x + i];
			if (tisstype != TISSUE_TYPE_AIR){
				float weight = 1;
				cog_x += (i + 0.5)*geometry.sample_dx*weight;
				cog_z += (k + 0.5)*geometry.sample_dz*weight;
				cog_tot += weight;
			}
		}
	}
	cerr << cog_x*1.0f/cog_tot << " " << cog_z*1.0f/cog_tot << endl;
	x0 = cog_x/cog_tot;
	z0 = cog_z/cog_tot;
	r0 = geometry.num_x*geometry.sample_dx;

	#endif
	

	//do allocation	
	float *A_device;
	cudaMalloc(&A_device, sizeof(float)*geometry.num_x*geometry.num_y*geometry.num_z);
	cudaMemcpy(A_device, A, sizeof(float)*geometry.num_x*geometry.num_y*geometry.num_z, cudaMemcpyHostToDevice);
	
	OpticalProps devOptProps; //cuda allocated optical properties
	createDeviceOptProps(&devOptProps, &optProps);

	//photon counts
	int num_steps = NUM_PHOTON_STEPS;//strtod(argv[5], NULL);//10;//45;//1;//strtod(argv[5], NULL);//45;
	int num_photons_per_packet = THREADS_PER_BLOCK*600;
	int started_photons = 0; //number of initialized photons
	
	//photon tracking arrays
	Photons photons;
	initializePhotons(&photons, num_photons_per_packet, ALLOC_GPU);

	Photons photons_backup;
	initializePhotons(&photons_backup, num_photons_per_packet, ALLOC_GPU);
	
	Photons hostPhotons;
	initializePhotons(&hostPhotons, num_photons_per_packet, ALLOC_HOST);
	
	Photons beamPhotons;
	initializePhotons(&beamPhotons, num_photons_per_packet, ALLOC_GPU);
	
	Photons beamPhotons_backup;
	initializePhotons(&beamPhotons_backup, num_photons_per_packet, ALLOC_GPU);
	
	Photons hostBeamPhotons;
	initializePhotons(&hostBeamPhotons, num_photons_per_packet, ALLOC_HOST);
	
	bool *photon_added = new bool[photons.num_photons](); //whether photon has been added to reflectance or not at the very end of the simulation
	
	//random number generator	
	RNGSeeds rngseeds;
	UINT64 seed = (UINT64) time(NULL);
	createRNGSeeds(&rngseeds, seed, num_photons_per_packet);
	cerr << "seed: " << seed << endl;

	//cuda thread distribution
	int block_number_y = 1;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int block_number_x = num_photons_per_packet/threadsPerBlock;
	dim3 dimGrid = dim3(block_number_x, block_number_y);
	dim3 dimBlock = dim3(threadsPerBlock);

	//Start MC simulation
	fprintf(stderr, "\nPerforming MC3D simulation using CUDA.\n");

	#ifndef WIN32
	timeval startTime, endTime;
	gettimeofday(&startTime, NULL);
	#endif

	bool finished = false; //whether simulation is finished
	bool firstTime = true; //first time photon_step is run

	cudaStream_t computeStream;
	cudaStream_t memcpyStream;
	cudaStreamCreate(&computeStream);
	cudaStreamCreate(&memcpyStream);

	int timesToPrint = num_photons/100000; //number of times we should print status updates

	int finished_photons_counted_from_gpu = 0;

	//GPU version

	//number of times the program is allowed not to move forward at the end
	int num_times_hangup = 0;
	int max_num_times_hangup = 50;
	const int PHOT_THRESHOLD = 2;
	bool saveBeam = true; //whether beam should be saved. Should continue one step past the point where we have stopped reinitializing
	int savedBeamPastNoReinit = 0; //number of times we have saved the beam past the point where we are doing no reinitializations
	int maxNumSavedBeamPastNoReinit = 2; //should be done at least once. But apparently also a second time. 


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
	#ifdef GPU_CALCREFL
	float *devRefl;
	cudaMalloc(&devRefl, sizeof(float)*geometry.num_x*geometry.num_y);
	float *devRefl_temphost = new float[geometry.num_x*geometry.num_y]();
	cudaMemcpy(devRefl, devRefl_temphost, sizeof(float)*geometry.num_x*geometry.num_y, cudaMemcpyHostToDevice);
	#endif

	//srand(time(NULL));
	int iteration = 0;
	while (!finished){
		/*if (rand()/(RAND_MAX*1.0f) < 0.001){
			fprintf(stderr, "GPU RNG was reinitialized...\n");
			seed = (UINT64) time(NULL);
			createRNGSeeds(&rngseeds, seed, num_photons_per_packet);
		}*/

		
		cudaEventRecord(start, 0);
	
		#ifndef GPU_CALCREFL
		cudaMemcpy(photons_backup.x, photons.x, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.y, photons.y, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.z, photons.z, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.W, photons.W, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.ux, photons.ux, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.uy, photons.uy, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.uz, photons.uz, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.s, photons.s, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		#endif

		cudaMemcpy(photons_backup.finished_photons, photons.finished_photons, sizeof(bool)*photons.num_photons, cudaMemcpyDeviceToDevice); 
		cudaMemcpy(photons_backup.initialized_photons, photons.initialized_photons, sizeof(int)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.num_finished_photons, photons.num_finished_photons, sizeof(int)*photons.num_photons, cudaMemcpyDeviceToDevice);
			
		
		if (saveBeam){
			cudaMemcpy(beamPhotons_backup.x, beamPhotons.x, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
			cudaMemcpy(beamPhotons_backup.y, beamPhotons.y, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
			cudaMemcpy(beamPhotons_backup.W, beamPhotons.W, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		}	
		
		
		if (started_photons < num_photons){
			#ifdef FINGER_SIMULATION
			photon_reinitialize<<<dimGrid, dimBlock, 0, computeStream>>>(geometry, photons, beamPhotons, rngseeds, optProps.n[TISSUE_TYPE_AIR], optProps.n[tissue_type_epidermis], x0, y0, z0, r0, arclength);
			#else
			photon_reinitialize<<<dimGrid, dimBlock, 0, computeStream>>>(geometry, photons, beamPhotons, rngseeds, optProps.n[TISSUE_TYPE_AIR], optProps.n[tissue_type_epidermis]);
			#endif
		}
		

		//run photons
		#ifdef GPU_CALCREFL
		photon_step<<<dimGrid, dimBlock, optProps.num_tissue_types*sizeof(float)*NUM_OPTPROPS, computeStream>>>(geometry, devOptProps, photons, rngseeds, num_steps, A_device, started_photons >= num_photons, devRefl);
		#else
		photon_step<<<dimGrid, dimBlock, optProps.num_tissue_types*sizeof(float)*NUM_OPTPROPS, computeStream>>>(geometry, devOptProps, photons, rngseeds, num_steps, A_device, started_photons >= num_photons);
		#endif
		
		cudaMemcpyAsync(hostPhotons.initialized_photons, photons_backup.initialized_photons, sizeof(int)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
		cudaMemcpyAsync(hostPhotons.num_finished_photons, photons_backup.num_finished_photons, sizeof(int)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);

		int num_prev_finished_photons = 0;
		
		if (!firstTime){	
			//device to device

			//download x,y,z,W to check if finished and save to reflectance arrays
			#ifndef GPU_CALCREFL
			cudaMemcpyAsync(hostPhotons.x, photons_backup.x, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.y, photons_backup.y, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.z, photons_backup.z, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.W, photons_backup.W, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.ux, photons_backup.ux, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.uy, photons_backup.uy, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.uz, photons_backup.uz, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.s, photons_backup.s, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			#endif
			cudaMemcpyAsync(hostPhotons.finished_photons, photons_backup.finished_photons, sizeof(bool)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream); //used when we are running again and again until all photons are frozen and finished


			if (saveBeam){
				cudaMemcpyAsync(hostBeamPhotons.x, beamPhotons_backup.x, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
				cudaMemcpyAsync(hostBeamPhotons.y, beamPhotons_backup.y, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
				cudaMemcpyAsync(hostBeamPhotons.W, beamPhotons_backup.W, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			}
		}
		

	
		//bookkeeping
			//start new photons until approximately the required amount of started photons
			//reinitialize photons

			//update reflectances with finished photons
			//save beam
			bool notFinished = false;
			if (!firstTime && (started_photons < num_photons)){
				notFinished = true;
			}
	
		num_prev_finished_photons = finished_photons_counted_from_gpu;


		int started_photons_newcount = 0;
		finished_photons_counted_from_gpu = 0;

		
		#ifndef GPU_CALCREFL
	
			#pragma omp parallel for	
			for (int i=0; i < photons.num_photons; i++){
				if (!firstTime && (started_photons < num_photons)){
					detector(i, geometry, hostPhotons, R, R_tot, T, T_tot);
					beam(i, geometry, hostBeamPhotons, B);
				} else if (!firstTime){
					//finish already started photons
					if (!hostPhotons.finished_photons[i]){
						notFinished = true;
					} else if (!photon_added[i]){
						detector(i, geometry, hostPhotons, R, R_tot, T, T_tot);
						photon_added[i] = true;
					}
					if (saveBeam){
						beam(i, geometry, hostBeamPhotons, B);
						savedBeamPastNoReinit++;
					}
				}
				
			}
			#endif

			for (int i=0; i < photons.num_photons; i++){	
				started_photons_newcount += hostPhotons.initialized_photons[i];
				finished_photons_counted_from_gpu += hostPhotons.num_finished_photons[i];
			}	
			
			if (!firstTime && !notFinished){
				finished = true;
			}


		/*for (int i=0; i < photons.num_photons; i++){
			if (!hostPhotons.finished_photons[i]){
				cout << "First photon not finished: " << i << " "  << hostPhotons.x[i] << " " << hostPhotons.y[i] << " " << hostPhotons.z[i] << " " << hostPhotons.ux[i] << " " << hostPhotons.uy[i] << " " << hostPhotons.uz[i] << " " << hostPhotons.s[i] << " " << hostPhotons.W[i] << endl;
				break;
			}
		}*/


		if ((started_photons >= num_photons) && savedBeamPastNoReinit > maxNumSavedBeamPastNoReinit){
			saveBeam = false;
		}

		started_photons = started_photons_newcount;

		if ((iteration % 1000) == 0){
		cout << started_photons << " " << finished_photons_counted_from_gpu << " " << finished_photons_counted_from_gpu - num_prev_finished_photons << " " << seed << endl;
		}
		iteration++;
		firstTime = false;
		

		#ifndef WIN32
		if (kill_process == true){
			finished = true;
		}


		#endif

		#if 0
		if (abs(started_photons - finished_photons_counted_from_gpu) < PHOT_THRESHOLD){
			num_times_hangup++;
			if (num_times_hangup > max_num_times_hangup){
				finished = true;
			}
		}
		#endif
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
	
		float time_cuda;
		cudaEventElapsedTime(&time_cuda, start, stop);
//		cerr << "Time used according to eventsyn: " <<  time_cuda << endl;
	}
	

	#ifdef GPU_CALCREFL
	cudaMemcpy(devRefl_temphost, devRefl, sizeof(float)*geometry.num_x*geometry.num_y, cudaMemcpyDeviceToHost);
	for (int i=0; i < geometry.num_x*geometry.num_y; i++){
		R[i] = devRefl_temphost[i];
	}
	#endif
	
	#ifndef WIN32
	gettimeofday(&endTime, NULL);
	float time = (endTime.tv_sec - startTime.tv_sec) + 1.0e-06*(endTime.tv_usec - startTime.tv_usec);
	cerr << "Time used: " <<  time << endl;
	#endif

	int num_phot_in_beam = 0;
	for (int i=0; i < geometry.num_x*geometry.num_y; i++){
		num_phot_in_beam += B[i];
	}
	if (num_phot_in_beam != started_photons){
		cerr << "Warning: Number of photons in beam is different from number of started photons: " << num_phot_in_beam << " " << started_photons << endl;
	}

	cudaMemcpy(A, A_device, sizeof(float)*geometry.num_x*geometry.num_y*geometry.num_z, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostPhotons.absorbed, photons.absorbed, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost);
	
	
	freeRNGSeeds(&rngseeds);
	cudaFreeArray(tissue_type_arr);

	
	deinitializePhotons(&hostPhotons);
	deinitializePhotons(&photons);
	
	cudaFree(A_device);

	delete [] photon_added;
	freeOptProps(&devOptProps);
	
	*num_finished_photons = finished_photons_counted_from_gpu;

}

