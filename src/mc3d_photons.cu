//=======================================================================================================
// Copyright 2015 Asgeir Bjorgan, Matija Milanic, Lise Lyngsnes Randeberg
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
// http://opensource.org/licenses/MIT)
//=======================================================================================================


#include "mc3d_types.h"
#include "mc3d_photons.h"
#include "mc3d_gpu.h"
#include <iostream>
using namespace std;

void photon_properties_initialize(photon_properties_t *photons, int num_photons, AllocType allocType){
	photons->allocWhere = allocType;
	switch (allocType){
		case ALLOC_HOST:
			photons->x = new float[num_photons]();
			photons->y = new float[num_photons]();
			photons->z = new float[num_photons]();
			
			photons->ux = new float[num_photons]();
			photons->uy = new float[num_photons]();
			photons->uz = new float[num_photons]();
			
			photons->W = new float[num_photons]();
			photons->s = new float[num_photons]();
			photons->absorbed = new float[num_photons]();

			photons->finished_photons = new bool[num_photons]();
			for (int i=0; i < num_photons; i++){
				photons->finished_photons[i] = 1;
			}
			
			photons->initialized_photons = new int[num_photons]();
			photons->num_finished_photons = new int[num_photons]();
			cerr << photons->initialized_photons << " " << photons->x << " address initialized photons" << endl;

			photons->num_photons = num_photons;
			
			photons->dummy = new float[num_photons]();
		break;

		case ALLOC_GPU:
			float *zero_arr = new float[num_photons]();
			int *zero_arr_int = new int[num_photons]();
			bool *zero_arr_bool = new bool[num_photons]();
			bool *ones_arr_bool = new bool[num_photons]();
			for (int i=0; i < num_photons; i++){
				ones_arr_bool[i] = true;
			}

			//initialize photon properties
			float *x, *y, *z, *ux, *uy, *uz, *s, *W;
			int *N, *num_finished_photons;
			bool *finished;
			cudaMalloc(&x, num_photons*sizeof(float));
			cudaMalloc(&y, num_photons*sizeof(float));
			cudaMalloc(&z, num_photons*sizeof(float));
			cudaMalloc(&N, num_photons*sizeof(int));
			cudaMalloc(&num_finished_photons, num_photons*sizeof(int));
			cudaMalloc(&(photons->dummy), num_photons*sizeof(int));
			
			cudaMalloc(&ux, num_photons*sizeof(float));
			cudaMalloc(&uy, num_photons*sizeof(float));
			cudaMalloc(&uz, num_photons*sizeof(float));
			
			cudaMalloc(&s, num_photons*sizeof(float));
			cudaMalloc(&W, num_photons*sizeof(float));
			cudaMalloc(&finished, num_photons*sizeof(bool));
			cudaMalloc(&(photons->absorbed), num_photons*sizeof(float));
			cudaMemcpy(photons->absorbed, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);

			cudaMemcpy(x, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(y, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(z, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);
			
			cudaMemcpy(photons->dummy, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);
			
			cudaMemcpy(ux, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(uy, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(uz, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);
			
			cudaMemcpy(s, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(W, zero_arr, num_photons*sizeof(float), cudaMemcpyHostToDevice);
			
			cudaMemcpy(N, zero_arr_int, num_photons*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(num_finished_photons, zero_arr_int, num_photons*sizeof(int), cudaMemcpyHostToDevice);
			
			cudaMemcpy(finished, ones_arr_bool, num_photons*sizeof(bool), cudaMemcpyHostToDevice);
			
			
			photons->x = x;
			photons->y = y;
			photons->z = z;

			photons->ux = ux;
			photons->uy = uy;
			photons->uz = uz;

			photons->W = W;
			photons->s = s;
			photons->num_photons = num_photons;
			photons->initialized_photons = N;
			photons->num_finished_photons = num_finished_photons;
			photons->finished_photons = finished;

			delete [] zero_arr;
			delete [] zero_arr_int;
			delete [] zero_arr_bool;
		break;
	}
}

void photon_properties_deinitialize(photon_properties_t *photons){
	switch (photons->allocWhere){
		case ALLOC_GPU:
			cudaFree(photons->x);
			cudaFree(photons->y);
			cudaFree(photons->z);
			cudaFree(photons->ux);
			cudaFree(photons->uy);
			cudaFree(photons->uz);
			cudaFree(photons->W);
			cudaFree(photons->s);
			cudaFree(photons->initialized_photons);
			cudaFree(photons->num_finished_photons);
			cudaFree(photons->finished_photons);
			cudaFree(photons->dummy);
		break;

		case ALLOC_HOST:
			delete [] photons->x;
			delete [] photons->y;
			delete [] photons->z;
			delete [] photons->ux;
			delete [] photons->uy;
			delete [] photons->uz;
			delete [] photons->W;
			delete [] photons->s;
			delete [] photons->initialized_photons;
			delete [] photons->num_finished_photons;
			delete [] photons->finished_photons;
			delete [] photons->dummy;
		break;
	}
}

//update R and T with the input photon number
void photon_detector(int photon, geometry_t geometry, photon_properties_t photons, double *R, double *totR, double *T, double *totT){
	if (photons.finished_photons[photon]){
		int ind_x = photontrack_get_grid_coord(&photons.x[photon], &photons.ux[photon], &geometry.sample_dx);
		int ind_y = photontrack_get_grid_coord(&photons.y[photon], &photons.uy[photon], &geometry.sample_dy);

		bool inside_geometry = ((ind_x >= 0) && (ind_y >= 0) && (ind_x < geometry.num_x) && (ind_y < geometry.num_y));
		if (photons.z[photon] <= 0){
			//add to reflectance
			if (inside_geometry){
				#pragma omp atomic
				R[ind_y*geometry.num_x + ind_x] += photons.W[photon];
			}
		} else {
			//add to transmission
			if (inside_geometry){
				#pragma omp atomic
				T[ind_y*geometry.num_x + ind_x] += photons.W[photon];
			} 
		}
	}
}
