#include "mc3d_types.h"
#include <string>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cstdio>
using namespace std;

int opticalprops_read_from_file(opticalprops_t *optProps, char *filename){
	FILE *stream;

	//read tissue types definition
	if ((stream = fopen(filename, "rb" )) == NULL){
		fclose(stream);
		fprintf(stderr, "The file %s was not opened\n", filename);
		return(-1);
	}

	//read no. of dimensions
	int dim;
	int numread = fread(&(optProps->num_tissue_types), sizeof(int), 1, stream);

	optProps->num_tissue_types += 1; //set tissue type position 0 to ambient medium

	//read optical properties
	float *mua = new float[optProps->num_tissue_types];
	float *mus = new float[optProps->num_tissue_types];
	float *g = new float[optProps->num_tissue_types];
	float *n = new float[optProps->num_tissue_types];
	double *temp = new double[NUM_OPTPROPS];

	//ambient medium
	mua[TISSUE_TYPE_AIR] = 0;
	mus[TISSUE_TYPE_AIR] = 0;
	g[TISSUE_TYPE_AIR] = 0;
	n[TISSUE_TYPE_AIR] = 1.0f;

	//the other tissue types
	for(int k=1; k < optProps->num_tissue_types; k++){
		numread = fread(temp, sizeof(double), NUM_OPTPROPS, stream);
		
		mua[k] = temp[0];
		mus[k] = temp[1];
		g[k] = temp[2];
		n[k] = temp[3];
	}
	delete [] temp;

	//close file
	fclose(stream);

	optProps->mua = mua;
	optProps->mus = mus;
	optProps->g = g;
	optProps->n = n;

	optProps->allocWhere = ALLOC_HOST;
	
	return 0;
}



void opticalprops_transfer_to_device(opticalprops_t *devOptProps, opticalprops_t *optProps){
	devOptProps->num_tissue_types = optProps->num_tissue_types;

	//allocate cuda arrays
	cudaMalloc(&(devOptProps->n), sizeof(float)*optProps->num_tissue_types);
	cudaMalloc(&(devOptProps->mua), sizeof(float)*optProps->num_tissue_types);
	cudaMalloc(&(devOptProps->mus), sizeof(float)*optProps->num_tissue_types);
	cudaMalloc(&(devOptProps->g), sizeof(float)*optProps->num_tissue_types);

	//copy to cuda arrays
	cudaMemcpy(devOptProps->n, optProps->n, sizeof(float)*optProps->num_tissue_types, cudaMemcpyHostToDevice);
	cudaMemcpy(devOptProps->mua, optProps->mua, sizeof(float)*optProps->num_tissue_types, cudaMemcpyHostToDevice);
	cudaMemcpy(devOptProps->mus, optProps->mus, sizeof(float)*optProps->num_tissue_types, cudaMemcpyHostToDevice);
	cudaMemcpy(devOptProps->g, optProps->g, sizeof(float)*optProps->num_tissue_types, cudaMemcpyHostToDevice);

	devOptProps->allocWhere = ALLOC_GPU;
}


void opticalprops_free(opticalprops_t *optProps){
	switch (optProps->allocWhere){
		case ALLOC_GPU:
			cudaFree(optProps->n);
			cudaFree(optProps->mua);
			cudaFree(optProps->mus);
			cudaFree(optProps->g);
		break;

		case ALLOC_HOST:
			delete [] optProps->n;
			delete [] optProps->g;
			delete [] optProps->mus;
			delete [] optProps->mua;
		break;
	}
}

int geometry_read_from_file(geometry_t *geometry, char *filename){
	FILE* stream;

	//open geometry file
	if ((stream = fopen(filename, "rb")) == NULL){
		fclose(stream);
		fprintf(stderr, "The file %s was not opened\n", filename);
		return -1;
	}
	
	//read no. of dimensions
	int dim;
	size_t numread = fread(&dim, sizeof(int), 1, stream);
	if (dim != 3){
		fclose(stream);
		fprintf(stderr, "Binary is not 3D.");
		return -1;
	}
	
	//read pixel numbers
	numread = fread(&(geometry->num_x), sizeof(int), 1, stream);
	numread = fread(&(geometry->num_y), sizeof(int), 1, stream);
	numread = fread(&(geometry->num_z), sizeof(int), 1, stream);
	
	//read pixel sizes
	double sample_dx, sample_dy, sample_dz; //in doubles
	numread = fread(&sample_dx, sizeof(double), 1, stream);
	numread = fread(&sample_dy, sizeof(double), 1, stream);
	numread = fread(&sample_dz, sizeof(double), 1, stream);

	//to float...
	geometry->sample_dx = sample_dx;
	geometry->sample_dy = sample_dy;
	geometry->sample_dz = sample_dz;

	//geometry sizes
	geometry->length_x = geometry->sample_dx*geometry->num_x;
	geometry->length_y = geometry->sample_dy*geometry->num_y;
	geometry->length_z = geometry->sample_dz*geometry->num_z;

	//read tissue type data
	geometry->tissue_type = new int[geometry->num_x*geometry->num_y*geometry->num_z];
	for (int k=0; k < geometry->num_z; k++){
		for (int j=0; j < geometry->num_y; j++){
			numread = fread(geometry->tissue_type + k*geometry->num_y*geometry->num_x + j*geometry->num_x, sizeof(int), geometry->num_x, stream);
		}
	}

	//increment tissue types by one so that tissue type no. 0 always will be ambient medium
	for (int i=0; i < geometry->num_x*geometry->num_y*geometry->num_z; i++){
		geometry->tissue_type[i] += 1;

		//switch from 255+1 to 0, since that would be air 
		if (geometry->tissue_type[i] == (255+1)){
			geometry->tissue_type[i] = TISSUE_TYPE_AIR;
		}
	}

	//close file
	fclose(stream);

	return 0;
}

void geometry_free(geometry_t *geometry){
	delete [] geometry->tissue_type;
}

void save_property(int numDims, geometry_t geometry, const double *data, const char *outName){
	if ((numDims < 2) || (numDims > 3)){
		fprintf(stderr, "Unsupported number of dimensions.\n");
		return;
	}


	FILE *stream;
	if ((stream = fopen(outName, "wb")) == NULL){
		fclose(stream);
		fprintf(stderr, "The file was not saved.\n");
		return;
	}

	size_t numsaved = fwrite(&numDims, sizeof(int), 1, stream);

	//array size
	numsaved = fwrite(&(geometry.num_x), sizeof(int), 1, stream);
	numsaved = fwrite(&(geometry.num_y), sizeof(int), 1, stream);
	if (numDims > 2){
		numsaved = fwrite(&(geometry.num_z), sizeof(int), 1, stream);
	}

	//voxel size
	double dx = geometry.sample_dx;
	double dy = geometry.sample_dy;
	double dz = geometry.sample_dz;
	numsaved = fwrite(&dx, sizeof(double), 1, stream);
	numsaved = fwrite(&dy, sizeof(double), 1, stream);
	if (numDims > 2){
		numsaved = fwrite(&dz, sizeof(double), 1, stream);
	}

	//write data
	int size = geometry.num_x*geometry.num_y;
	if (numDims > 2){
		size *= geometry.num_z;
	}
	numsaved = fwrite(data, sizeof(double), size, stream);

	fclose(stream);
}

void save_diff_refl(geometry_t geometry, double *diffRefl, int numPhotons, char *outName){
	string filename = string(outName) + "_drs.bin";
	double *diffReflSaved = new double[geometry.num_x*geometry.num_y];
	for (int i=0; i < geometry.num_x*geometry.num_y; i++){
		diffReflSaved[i] = diffRefl[i]/(numPhotons*1.0);
	}

	saveProperty(2, geometry, diffReflSaved, filename.c_str());
	delete [] diffReflSaved;
}

void save_abs_map(geometry_t geometry, opticalprops_t optProps, float *abs, int numPhotons, char *outName){
	//absorption map
	string filename = string(outName) + "_abs.bin";
	double *absDbl = new double[geometry.num_x*geometry.num_y*geometry.num_z];

	double beamEnergy = 1.0; //FIXME
	double normalization = numPhotons*1.0;//beamEnergy*geometry.length_x*geometry.length_y/(numPhotons*geometry.sample_dx*geometry.sample_dy*geometry.sample_dz); 

	for (int i=0; i < geometry.num_x*geometry.num_y*geometry.num_z; i++){
		absDbl[i] = abs[i]*normalization;
	}
	saveProperty(3, geometry, absDbl, filename.c_str());

	//fluence map
	for (int k=0; k < geometry.num_z; k++){
		for (int j=0; j < geometry.num_y; j++){
			for (int i=0; i < geometry.num_x; i++){
				int ind = k*geometry.num_y*geometry.num_x + j*geometry.num_x + i;
				int tissue_type = geometry.tissue_type[ind];
				absDbl[ind] /= optProps.mua[tissue_type];
			}
		}
	}
	filename = string(outName) + "_flu.bin";
	saveProperty(3, geometry, absDbl, filename.c_str());

	delete [] absDbl;
}
