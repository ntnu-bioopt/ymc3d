#include <cmath>
#include "mc3d_rng.cu"

#ifndef WIN32
#include <sys/time.h>
#include <signal.h>
#endif

#include <iostream>
#include "mc3d_types.h"
#include "mc3d_io.h"
#include "mc3d_photons.h"
#include <cstdio>
using namespace std;

texture<int, 3, cudaReadModeElementType> tissue_type_tex; //3D texture containing spatially resolved tissue types
__device__ int photontrack_get_tissue_type(int i, int j, int k, float z, float uz){
	int tissue_type = tex3D(tissue_type_tex, i, j, k);
	return ((z <= 0) && (uz < 0)) ? TISSUE_TYPE_AIR : tissue_type;
}

__device__ float photontrack_intersection(geometry_t geometry, int *i, int *j, int *k, float *x, const float *ux, float *y, const float *uy, float *z, const float *uz){
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
__host__ __device__ int photontrack_get_grid_coord(const float *x, const float *ux, const float *sample_dx){
	int i = floorf(*x/(*sample_dx)); //initial estimate. Assumes that the coordinate is either within bounding box or on the boundary with positive direction. Rounds down
	i += (*x == (i+1)*(*sample_dx)); //quickfix numerical errors
	float bmin = i*(*sample_dx);

	//check whether position is on the bmin boundaries (will never be on the bmax boundaries since that would be the bmin of the next box
	//if it does, determine whether indices should stay the same or decremented according to the direction
	i = ((*x > bmin) || ((*x == bmin) && (*ux >= 0))) ? i : i-1;

	return i;
}


//calculate specular reflection
__device__ float photontrack_calc_specular_reflection(float n0, float n1){
	float Rsp = (n0 - n1)/(n0 + n1);
	return Rsp*Rsp;
}


//n0 is refraction index of ambient medium, n1 is refraction index of upper voxels. Assume they won't vary spatially for simplicity
__global__ void photontrack_reinitialize(geometry_t geometry, photon_properties_t photons, rng_state_t rngseeds, float n0, float n1){
	int ind = (blockIdx.x + blockIdx.y*gridDim.x)*blockDim.x + blockDim.x*threadIdx.y + threadIdx.x;
	float newx = 0;
	float newy = 0;
	float newz = 0;
	float Rsp = 2;
	if (photons.finished_photons[ind]){
		//initialize new photon if the former has frozen
		UINT64 x = rngseeds.rng_x[ind];
		UINT32 a = rngseeds.rng_a[ind];
		
		float newux = 0, newuy = 0, newuz = 1;

		#if 1
		//uniform illumination
		newx = geometry.length_x*rand_MWC_co(&x, &a);
		newy = geometry.length_y*rand_MWC_co(&x, &a);
		#else
		//beam
		float newx = geometry.length_x/2;
		float newy = geometry.length_y/2;
		#endif

		photons.x[ind] = newx, photons.y[ind] = newy, photons.z[ind] = newz;

		Rsp = photontrack_calc_specular_reflection(n0, n1);
		photons.ux[ind] = newux, photons.uy[ind] = newuy, photons.uz[ind] = newuz;
		photons.W[ind] = 1.0f - Rsp;
		photons.s[ind] = -log(rand_MWC_oc(&x, &a));
		(photons.initialized_photons[ind])++;

		rngseeds.rng_x[ind] = x;
		rngseeds.rng_a[ind] = a;
		
		photons.finished_photons[ind] = false;
		
	} 
}			

__device__ bool photontrack_mirror(float *x, float *ux, const float *length_x){
	bool isReflected = (*x >= *length_x) || (*x <= 0);
	
	//x > length
	*x = (*x >= *length_x) ? 2*(*length_x) - *x : *x;


	//x < 0
	*x = (*x <= 0) ? fabsf(*x) : *x;

	*ux = (isReflected) ? -1*(*ux) : *ux;

	return isReflected;
}

__global__ void photontrack_step(geometry_t geometry, opticalprops_t optProps, photon_properties_t photons, rng_state_t rngseeds, int steps, float *A, bool shouldBreak){
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
		
	int i = photontrack_get_grid_coord(&x, &ux, &geometry.sample_dx);
	int j = photontrack_get_grid_coord(&y, &uy, &geometry.sample_dy);
	int k = photontrack_get_grid_coord(&z, &uz, &geometry.sample_dz);
	int tissue_type = photontrack_get_tissue_type(i, j, k, z, uz); //current tissue type

	float db;

	for (int step = 0; step < steps; step++){
		float newx, newy, newz;
		if (shouldBreak && freeze){
			break;
		}
		
		float ni = sh_n[tissue_type];
		float nt = ni;
		float epsilon;

		float mu_a = sh_mua[tissue_type];
		float mu_s = sh_mus[tissue_type];
		float g = sh_g[tissue_type];
		float mu_t = mu_a + mu_s;

		//estimate distance to boundary of the box in particle direction
		newx = x;
		newy = y;
		newz = z;
		db = photontrack_intersection(geometry, &i, &j, &k, &newx, &ux, &newy, &uy, &newz, &uz); //provided ux,uy,uz normalized: db is distance to boundary

		//find distance to move
		bool hit_boundary = db*mu_t <= s;
		float ds = hit_boundary ? db*mu_t : s; //travel distance s if boundary is further away, otherwise travel distance to boundary

		//calculate new coordinates
		newx = (hit_boundary) ? newx : x + ux*ds/mu_t;
		newy = (hit_boundary) ? newy : y + uy*ds/mu_t;
		newz = (hit_boundary) ? newz : z + uz*ds/mu_t;

		
		//update s
		s = s - ds;
		
		int nexti = photontrack_get_grid_coord(&newx, &ux, &geometry.sample_dx);
		int nextj = photontrack_get_grid_coord(&newy, &uy, &geometry.sample_dy);
		int nextk = photontrack_get_grid_coord(&newz, &uz, &geometry.sample_dz);
		
		int next_tissue_type = photontrack_get_tissue_type(nexti, nextj, nextk, newz, uz);
		nt = sh_n[next_tissue_type];
		
		float scatt_ux, scatt_uy, scatt_uz;
		float transrefl_ux = ux, transrefl_uy = uy, transrefl_uz = uz;

		bool isReflected = false;
	
		//TRANSMIT/REFLECT
		
		//calculate incident angle on the assumption that photon has passed into the next cell
		
		//for the most of the threads, ni is going to be equal to nt, so we might reduce thread divergence by checking for that.
		if (ni != nt){ //if a single thread does not have ni != nt, however, all threads will run the next code segment and waste time on nothing, but in case we are lucky it will help. 
			float cos_ai = (i != nexti) ? ux : ((j != nextj) ? uy : uz);
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
			float R2 = (cos_ai*cos_at - sin_ai*sin_at)/(cos_ai*cos_at + sin_at*sin_at);
			float R = 0.5*R1*R1*(1.0f + R2*R2);

			R = (sin_at > 1) ? 1 : R; //critical angle
			R = (ni == nt) ? 0 : R;

			epsilon = rand_MWC_co(&rnd_x, &rnd_a);
			isReflected = epsilon < R;

			//decide new direction based on assumption of reflection/transmission, and which cell the photon passed through

			//reflection: negative sign if reflected in primary direction, otherwise the same
			//transmission: u*ni/nt if not primary direction, signum(u)*cosf(at) if primary direction
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
		isReflected = isReflected || photontrack_mirror(&x, &ux, &geometry.length_x) || photontrack_mirror(&y, &uy, &geometry.length_y);
		#endif
	
		//update grid coordinates
		i = (!((hit_boundary && isReflected) || freeze)) ? nexti : i;
		j = (!((hit_boundary && isReflected) || freeze)) ? nextj : j;
		k = (!((hit_boundary && isReflected) || freeze)) ? nextk : k;

	
		tissue_type = (!((hit_boundary && isReflected) || freeze)) ? next_tissue_type : tissue_type;
	}

	//quickfix in case anything goes horribly wrong. 
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

	//increment number of finished photons
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
	
	rngseeds.rng_a[ind] = rnd_a;
	rngseeds.rng_x[ind] = rnd_x;
}


#ifndef WIN32
bool kill_process = false;
void sig_handler(int signo){
	if (signo == SIGINT){
		kill_process = true;
	}
}
#endif

void run_3dmc_gpu(geometry_t geometry, opticalprops_t optProps, int num_photons, double *R, double *R_tot, double *T, double *T_tot, float *A, int *num_finished_photons){
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

	//do allocation	
	float *A_device;
	cudaMalloc(&A_device, sizeof(float)*geometry.num_x*geometry.num_y*geometry.num_z);
	cudaMemcpy(A_device, A, sizeof(float)*geometry.num_x*geometry.num_y*geometry.num_z, cudaMemcpyHostToDevice);
	
	opticalprops_t devOptProps; //cuda allocated optical properties
	opticalprops_transfer_to_device(&devOptProps, &optProps);

	//photon counts
	int num_steps = NUM_PHOTON_STEPS;
	int num_photons_per_packet = THREADS_PER_BLOCK*600;
	int started_photons = 0; //number of initialized photons
	
	//photon tracking arrays
	photon_properties_t photons;
	photon_properties_initialize(&photons, num_photons_per_packet, ALLOC_GPU);

	photon_properties_t photons_backup;
	photon_properties_initialize(&photons_backup, num_photons_per_packet, ALLOC_GPU);
	
	photon_properties_t hostPhotons;
	photon_properties_initialize(&hostPhotons, num_photons_per_packet, ALLOC_HOST);
	
	photon_properties_t hostBeamPhotons;
	photon_properties_initialize(&hostBeamPhotons, num_photons_per_packet, ALLOC_HOST);
	
	bool *photon_added = new bool[photons.num_photons](); //whether photon has been added to reflectance or not at the very end of the simulation
	
	//random number generator	
	rng_state_t rngseeds;
	UINT64 seed = (UINT64) time(NULL);
	rng_init(&rngseeds, seed, num_photons_per_packet);
	cerr << "seed: " << seed << endl;

	//cuda thread distribution
	int block_number_y = 1;
	int threadsPerBlock = THREADS_PER_BLOCK;
	int block_number_x = num_photons_per_packet/threadsPerBlock;
	dim3 dimGrid = dim3(block_number_x, block_number_y);
	dim3 dimBlock = dim3(threadsPerBlock);

	//Start MC simulation
	fprintf(stderr, "\nPerforming MC3D simulation using CUDA.\n");

	bool finished = false; //whether simulation is finished
	bool firstTime = true; //first time photon_step is run

	cudaStream_t computeStream;
	cudaStream_t memcpyStream;
	cudaStreamCreate(&computeStream);
	cudaStreamCreate(&memcpyStream);

	int finished_photons_counted_from_gpu = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	int iteration = 0;
	while (!finished){
		cudaEventRecord(start, 0);
	
		cudaMemcpy(photons_backup.x, photons.x, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.y, photons.y, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.z, photons.z, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.W, photons.W, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.ux, photons.ux, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.uy, photons.uy, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.uz, photons.uz, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.s, photons.s, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToDevice);

		cudaMemcpy(photons_backup.finished_photons, photons.finished_photons, sizeof(bool)*photons.num_photons, cudaMemcpyDeviceToDevice); 
		cudaMemcpy(photons_backup.initialized_photons, photons.initialized_photons, sizeof(int)*photons.num_photons, cudaMemcpyDeviceToDevice);
		cudaMemcpy(photons_backup.num_finished_photons, photons.num_finished_photons, sizeof(int)*photons.num_photons, cudaMemcpyDeviceToDevice);
			
		if (started_photons < num_photons){
			photontrack_reinitialize<<<dimGrid, dimBlock, 0, computeStream>>>(geometry, photons, rngseeds, optProps.n[TISSUE_TYPE_AIR], optProps.n[tissue_type_epidermis]);
		}

		//run photons
		photontrack_step<<<dimGrid, dimBlock, optProps.num_tissue_types*sizeof(float)*NUM_OPTPROPS, computeStream>>>(geometry, devOptProps, photons, rngseeds, num_steps, A_device, started_photons >= num_photons);
		
		cudaMemcpyAsync(hostPhotons.initialized_photons, photons_backup.initialized_photons, sizeof(int)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
		cudaMemcpyAsync(hostPhotons.num_finished_photons, photons_backup.num_finished_photons, sizeof(int)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);

		int num_prev_finished_photons = 0;
		
		if (!firstTime){	
			//device to device

			//download x,y,z,W to check if finished and save to reflectance arrays
			cudaMemcpyAsync(hostPhotons.x, photons_backup.x, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.y, photons_backup.y, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.z, photons_backup.z, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.W, photons_backup.W, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.ux, photons_backup.ux, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.uy, photons_backup.uy, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.uz, photons_backup.uz, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.s, photons_backup.s, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream);
			cudaMemcpyAsync(hostPhotons.finished_photons, photons_backup.finished_photons, sizeof(bool)*photons.num_photons, cudaMemcpyDeviceToHost, memcpyStream); //used when we are running again and again until all photons are frozen and finished
		}
		
		//bookkeeping
		bool notFinished = false;
		if (!firstTime && (started_photons < num_photons)){
			notFinished = true;
		}
	
		num_prev_finished_photons = finished_photons_counted_from_gpu;


		int started_photons_newcount = 0;
		finished_photons_counted_from_gpu = 0;

		
		#pragma omp parallel for	
		for (int i=0; i < photons.num_photons; i++){
			if (!firstTime && (started_photons < num_photons)){
				photon_detector(i, geometry, hostPhotons, R, R_tot, T, T_tot);
			} else if (!firstTime){
				//finish already started photons
				if (!hostPhotons.finished_photons[i]){
					notFinished = true;
				} else if (!photon_added[i]){
					photon_detector(i, geometry, hostPhotons, R, R_tot, T, T_tot);
					photon_added[i] = true;
				}
			}
			
		}

		for (int i=0; i < photons.num_photons; i++){	
			started_photons_newcount += hostPhotons.initialized_photons[i];
			finished_photons_counted_from_gpu += hostPhotons.num_finished_photons[i];
		}	
			
		if (!firstTime && !notFinished){
			finished = true;
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

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
	
		float time_cuda;
		cudaEventElapsedTime(&time_cuda, start, stop);
	}
	
	cudaMemcpy(A, A_device, sizeof(float)*geometry.num_x*geometry.num_y*geometry.num_z, cudaMemcpyDeviceToHost);
	cudaMemcpy(hostPhotons.absorbed, photons.absorbed, sizeof(float)*photons.num_photons, cudaMemcpyDeviceToHost);
	
	rng_free(&rngseeds);
	cudaFreeArray(tissue_type_arr);
	
	photon_properties_deinitialize(&hostPhotons);
	photon_properties_deinitialize(&photons);
	
	cudaFree(A_device);

	delete [] photon_added;
	opticalprops_free(&devOptProps);
	
	*num_finished_photons = finished_photons_counted_from_gpu;
}

