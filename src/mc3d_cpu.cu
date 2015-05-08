#include "mc3d_photons.h"
#include "mc3d_gpu.h"
#include "mc3d_types.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
using namespace std;

int getTissueType_host(Geometry geometry, int i, int j, int k, float z, float uz){
	int tissue_type = 0;

	int tissue_i = i, tissue_j = j, tissue_k = k;
	if (i < 0){
		tissue_i = 0;
	}
	if (i >= geometry.num_x){
		tissue_i = geometry.num_x - 1;
	}
	if (j < 0){
		tissue_j = 0;
	}
	if (j >= geometry.num_y){
		tissue_j = geometry.num_y - 1;
	}
	if (k < 0){
		tissue_k = 0;
	}
	if (k >= geometry.num_z){
		tissue_k = geometry.num_z - 1;
	}

	tissue_type = geometry.tissue_type[tissue_k*geometry.num_x*geometry.num_y + tissue_j*geometry.num_x + tissue_i];
	return ((z <= 0) && (uz < 0)) ? 0 : tissue_type;
}

float getRand(){
	return rand()/(1.0f*RAND_MAX);
}

void photon_reinitialize_host(int ind, Geometry geometry, Photons photons, float n0, float n1){
	if (photons.finished_photons[ind]){
		//initialize new photon if the former has frozen
		photons.x[ind] = geometry.length_x/2, photons.y[ind] = geometry.length_y/2, photons.z[ind] = 0; //beam

		float Rsp = calcRSp(n0, n1);
		photons.ux[ind] = 0, photons.uy[ind] = 0, photons.uz[ind] = 1;
		photons.W[ind] = 1.0f - Rsp;
		photons.s[ind] = -log(getRand());
		(photons.initialized_photons[ind])++;

		
		photons.finished_photons[ind] = false;
	}

	//write shared memory reflectance back to global memory reflectance
}			

float host_intersection(Geometry geometry, int *i, int *j, int *k, float *x, const float *ux, float *y, const float *uy, float *z, const float *uz){
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

void photon_step_host(int ind, Geometry geometry, OpticalProps optProps, Photons photons, int steps, float *A, float *A_tot){
	//read from arrays
	float x = photons.x[ind];
	float y = photons.y[ind];
	float z = photons.z[ind];
	
	float ux = photons.ux[ind];
	float uy = photons.uy[ind];
	float uz = photons.uz[ind];

	float s = photons.s[ind];
	float W = photons.W[ind];

		
	//start grid coordinates
	int i = getGridCoord(&x, &ux, &geometry.sample_dx);
	int j = getGridCoord(&y, &uy, &geometry.sample_dy);
	int k = getGridCoord(&z, &uz, &geometry.sample_dz);

	float newx, newy, newz;


	bool freeze = photons.finished_photons[ind];
	if (freeze){
		return;
	}
	bool wasFrozen = freeze;
	int tissue_type = getTissueType_host(geometry, i, j, k, z, uz); //current tissue type

	for (int step = 0; step < steps; step++){

		int i = getGridCoord(&x, &ux, &geometry.sample_dx);
		int j = getGridCoord(&y, &uy, &geometry.sample_dy);
		int k = getGridCoord(&z, &uz, &geometry.sample_dz);
		int tissue_type = getTissueType_host(geometry, i, j, k, z, uz);
		
		float ni = optProps.n[tissue_type];
		float nt = ni;
		float epsilon;

		//FIXME: mu_a and mu_t read from the current cell (not next cell)
		float mu_a = optProps.mua[tissue_type];
		float mu_s = optProps.mus[tissue_type];
		float g = optProps.g[tissue_type];
		float mu_t = mu_a + mu_s;

		//bounding box coordinates
		float bmin_x = i*geometry.sample_dx;
		float bmin_y = j*geometry.sample_dy;
		float bmin_z = k*geometry.sample_dz;
		float bmax_x = (i+1)*geometry.sample_dx;
		float bmax_y = (j+1)*geometry.sample_dy;
		float bmax_z = (k+1)*geometry.sample_dz;

		//estimate distance to boundary of the box in particle direction
		newx = x;
		newy = y;
		newz = z;
		float db = host_intersection(geometry, &i, &j, &k, &newx, &ux, &newy, &uy, &newz, &uz); //provided ux,uy,uz normalized: db is distance to boundary
		//db = (db == 0) ? 2*s/mu_t : db; //numerical inaccuracies or ambiguous grid coordinates can cause db to be zero, will freeze forever. Just set it to twice the stepsize in that case... Also set it to twice the size of the photon happens to be outside the volume of interest
		
		//undefined if pointing along grid rod

		//find distance to move
		bool hit_boundary = db*mu_t <= s;
		float ds = hit_boundary ? db*mu_t : s; //travel distance s if boundary is further away, otherwise travel distance to boundary

		//calculate new coordinates
		newx = (hit_boundary) ? newx : x + ux*ds/mu_t;
		newy = (hit_boundary) ? newy : y + uy*ds/mu_t;
		newz = (hit_boundary) ? newz : z + uz*ds/mu_t;
	

		//update s
		s = s - ds;

		//correct for precision errors when coordinates are close to zero: depending on the situation, will get closer and closer to zero but never truly converge
		//newx = (fabs(newx) < ZERO_THRESH) ? 0 : newx;
		//newy = (fabs(newy) < ZERO_THRESH) ? 0 : newy;
		//newz = (fabs(newz) < ZERO_THRESH) ? 0 : newz;
		
		int nexti = getGridCoord(&newx, &ux, &geometry.sample_dx);
		int nextj = getGridCoord(&newy, &uy, &geometry.sample_dy);
		int nextk = getGridCoord(&newz, &uz, &geometry.sample_dz);
		

		int next_tissue_type = getTissueType_host(geometry, nexti, nextj, nextk, newz, uz);
		nt = optProps.n[next_tissue_type];

		
		float scatt_ux, scatt_uy, scatt_uz;
		float transrefl_ux = ux, transrefl_uy = uy, transrefl_uz = uz;

		//TRANSMIT/REFLECT

		//calculate incident angle on the assumption that photon has passed into the next cell
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


		epsilon = getRand();
		bool isReflected = epsilon <= R;

		//decide new direction based on assumption of reflection/transmission, and which cell the photon passed through

		//reflection: negative sign if reflected in primary direction, otherwise the same
		//transmission: u*ni/nt if not primary direction, signum(u)*cosf(at) if primary direction
		//NB: preference ordering! If ux was chosen as primary direction, uy and uz should not be treated as primary angles. They are changed only if the previous wasn't changed..
		transrefl_ux = (isReflected) ? ux*((i == nexti) ? 1 : -1) : ((i != nexti) ? copysignf(cos_at, ux*cos_at) : ux*ni_nt);
		transrefl_uy = (isReflected) ? uy*((j == nextj) ? 1 : -1) : (((j != nextj) && (i == nexti)) ? copysignf(cos_at, uy*cos_at) : uy*ni_nt); 
		transrefl_uz = (isReflected) ? uz*((k == nextk) ? 1 : -1) : (((k != nextk) && (j == nextj) && (i == nexti)) ? copysignf(cos_at, uz*cos_at) : uz*ni_nt);
		
		transrefl_ux = (ni == nt) ? ux : transrefl_ux;
		transrefl_uy = (ni == nt) ? uy : transrefl_uy;
		transrefl_uz = (ni == nt) ? uz : transrefl_uz;

		
		//ABSORPTION

		//absorption calculation
		float dW = (mu_t > 0) ? mu_a/mu_t*W : 0; //will be zero if we still are in an ambient medium
		W = W - dW*(!hit_boundary); //do an absorption if we ain't hittin' any boundaries, and aren't in an ambient medium
		if ((!hit_boundary)){
			int abs_i = i, abs_j = j, abs_k = k;
			

			if ((abs_i < 0) || (abs_i >= geometry.num_x)){
				abs_i = 0;
			}
			if ((abs_j < 0) || (abs_j >= geometry.num_y)){
				abs_j = 0;
			}
			if ((abs_k < 0) || (abs_k >= geometry.num_z)){
				abs_k = 0;
			}
			

			//int abs_i = ((i >= 0) && (i < geometry.num_x)) ? i : ((i < 0) ? 0 : geometry.num_x - 1);
			//int abs_j = ((j >= 0) && (j < geometry.num_y)) ? j : ((j < 0) ? 0 : geometry.num_y - 1);
			//int abs_k = ((k >= 0) && (k < geometry.num_z)) ? k : ((k < 0) ? 0 : geometry.num_z - 1);


			A[abs_k*geometry.num_y*geometry.num_x + abs_j*geometry.num_x + abs_i] += dW;
			*A_tot = *A_tot + dW;

			photons.absorbed[ind] += dW;
		}

		//russian roulette
		epsilon = getRand();
		float newW = (epsilon <= SURVIVAL_CHANCE) ? W/SURVIVAL_CHANCE : 0;
		W = ((W < W_THRESHOLD) && (!hit_boundary) && (mu_t != 0)) ? newW : W;

		//SCATTERING

		//decide new direction based on assumption of scattering
		epsilon = getRand();
		float costheta = 1.0f/(2*g)*(1 + g*g - (1-g*g)/(1-g+2*g*epsilon)*(1-g*g)/(1-g+2*g*epsilon));
		float sintheta = sqrtf(1-costheta*costheta);
		epsilon = getRand();
		float phi = 2*M_PI*epsilon;

		float cosphi = cosf(phi);
		float sinphi = sqrtf(1-cosphi*cosphi);
		float invsqrt = 1.0f/sqrtf(1-uz*uz);

		//avoid division by small number when uz > 0.99999
		scatt_ux = (fabsf(uz) <= UZ_THRESH) ? sintheta*(ux*uz*cosphi - uy*sinphi)*invsqrt + ux*costheta : sintheta*cosphi;
		scatt_uy = (fabsf(uz) <= UZ_THRESH) ? sintheta*(uy*uz*cosphi + ux*sinphi)*invsqrt + uy*costheta : sintheta*sinphi;
		scatt_uz = (fabsf(uz) <= UZ_THRESH) ? -1*sqrtf(1-uz*uz)*sintheta*cosphi + uz*costheta : copysignf(costheta, uz*costheta);

		//new direction based on actual events
		float newux = (!hit_boundary) ? scatt_ux : transrefl_ux;
		float newuy = (!hit_boundary) ? scatt_uy : transrefl_uy;
		float newuz = (!hit_boundary) ? scatt_uz : transrefl_uz;
		
		float zero_thresh = 1 - UZ_THRESH;
		newux = (fabs(newux) < zero_thresh) ? 0 : newux; 
		newuy = (fabs(newuy) < zero_thresh) ? 0 : newuy; 
		newuz = (fabs(newuz) < zero_thresh) ? 0 : newuz; 
		
		float newu_norm = 1.0f/sqrtf(newux*newux + newuy*newuy + newuz*newuz);
		newux *= newu_norm;
		newuy *= newu_norm;
		newuz *= newu_norm;

		//actual new direction based on dead or not and ambient medium (i.e. keep old direction forever if photon is dead or has escaped)
		freeze = (W <= 0) || (mu_t == 0);
		ux = (!freeze) ? newux : ux;
		uy = (!freeze) ? newuy : uy;
		uz = (!freeze) ? newuz : uz;


		//set new s if necessary
		epsilon = getRand();
		float newS = (W > 0) ? -log(epsilon) : 0;
		s = (freeze || hit_boundary) ? s : newS;

		//set new final coordinates
		x = (!freeze) ? newx : x;
		y = (!freeze) ? newy : y;
		z = (!freeze) ? newz : z;
		
		
	
		//update grid coordinates
		i = (!((hit_boundary && isReflected) || freeze)) ? nexti : i;
		j = (!((hit_boundary && isReflected) || freeze)) ? nextj : j;
		k = (!((hit_boundary && isReflected) || freeze)) ? nextk : k;


	
		tissue_type = (!((hit_boundary && isReflected) || freeze)) ? next_tissue_type : tissue_type;

		if (freeze){
			break;
		}
	}	

	if ((freeze != wasFrozen) && (freeze)){
		photons.num_finished_photons[ind]++;
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
	
}


void run_3dmc_cpu(Geometry geometry, OpticalProps optProps, int num_photons, double *R, double *R_tot, double *T, double *T_tot, float *A, int *num_finished_photons){
	bool finished = false;
	int started_photons = 0;
	int finished_photons = 0;

	Photons photons;
	int num_photons_per_packet = 300;
	int num_steps = NUM_PHOTON_STEPS;
	initializePhotons(&photons, num_photons_per_packet, ALLOC_HOST);
	bool *photon_added = new bool[photons.num_photons](); //whether photon has been added to reflectance or not at the very end of the simulation

	float A_tot_gpucalc = 0;
	//CPU version
	while (!finished){
		if (started_photons < num_photons){
			for (int ind = 0; ind < photons.num_photons; ind++){
				photon_reinitialize_host(ind, geometry, photons, optProps.n[0], optProps.n[1]);
			}
		}
		
		//run photons
		for (int ind = 0; ind < photons.num_photons; ind++){
			photon_step_host(ind, geometry, optProps, photons, num_steps, A, &A_tot_gpucalc);
		}	
		
		//bookkeeping
		if (started_photons < num_photons){
			for (int i=0; i < photons.num_photons; i++){
				detector(i, geometry, photons, R, R_tot, T, T_tot);
			}
		} else {
			//finish already started photons
			bool notFinished = false;
			for (int i=0; i < photons.num_photons; i++){
				if (!photons.finished_photons[i]){
					notFinished = true;
				} else if (!photon_added[i]){
					detector(i, geometry, photons, R, R_tot, T, T_tot);
					photon_added[i] = true;
				}
			}

			if (!notFinished){
				finished = true;
			}
		}

		//control number of started photons
		started_photons = 0;
		finished_photons = 0;
		for (int i=0; i < photons.num_photons; i++){
			started_photons += photons.initialized_photons[i];
			finished_photons += photons.num_finished_photons[i];
		}
		cerr << started_photons << " " << finished_photons << endl;
	}
	*num_finished_photons = finished_photons;

	deinitializePhotons(&photons);
}	
