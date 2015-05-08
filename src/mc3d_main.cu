#include "mc3d_gpu.h"
#include "mc3d_cpu.h"
#include "mc3d_types.h"
#include "mc3d_io.h"
#include <cstdio>
#include <iostream>
using namespace std;

int main(int argc, char* argv[]){
	//argument parsing
	char *geomPath, *tissuePath, *defPath, *outName;
	if (argc < 5){
		fprintf(stderr, "\nUsage: %s geometry.bin tissue.bid definition.txt OutputName\n", argv[0]);
		return 1;
	}
	geomPath = (char*)argv[1];
	tissuePath = (char*)argv[2];
	defPath = (char*)argv[3];
	outName = (char*)argv[4];

	//read geometry and tissue definitions
	geometry_t geometry;
	int stat = geometry_read_from_file(&geometry, geomPath); //includes cuda allocated tissue texture array
	opticalprops_t optProps;
	stat += opticalprops_read_from_file(&optProps, tissuePath);
	if (stat != 0){
		return 1;
	}
	fprintf(stderr, "\nRead geometry: %s and tissue definition: %s.\n",geomPath,tissuePath);

	//print optical properties to standard output
	cout << "#mua (mm-1), mus (mm-1), g, n. Rows: tissue type 1, tissue type 2, ..." << endl;
	for (int i=1; i < optProps.num_tissue_types; i++){
		cout << optProps.mua[i] << " " << optProps.mus[i] << " " << optProps.g[i] << " " << optProps.n[i] << endl;
	}

	//diffuse reflectance
	double *R = new double[geometry.num_x*geometry.num_y]();
	double totDiffR = 0; //reflectance both inside and outside geometry

	//transmittance
	double *T = new double[geometry.num_x*geometry.num_y]();
	double totT = 0; //transmittance both inside and outside geometry

	//absorption map
	float *A = new float[geometry.num_x*geometry.num_y*geometry.num_z]();
	
	//photon book keeping

	//photon numbers
	int num_photons = 30000000;
	int finished_photons = 0; //number of finished photons

	//run simulation	
	run_3dmc_gpu(geometry, optProps, num_photons, R, &totDiffR, T, &totT, A, &finished_photons);

	//Calculate diffuse reflectance
	fprintf(stderr, "Total diffuse reflectance is %f.\n", totDiffR/finished_photons);
	float geomDiffR = 0; //total diffuse reflectance over geometry area
	for (int i=0; i < geometry.num_x*geometry.num_y; i++){
		geomDiffR += R[i];
	}
	fprintf(stderr, "(diffuse reflectance over geometry area is %f)\n", geomDiffR/finished_photons);

	//calculate total absorption
	double totAbsIn = 0;
	for (int i=0; i < geometry.num_x*geometry.num_y*geometry.num_z; i++){
		totAbsIn += A[i];
	}
	
	fprintf(stderr, "Absorbed fraction in geometry is %f.\n", totAbsIn/finished_photons);
	fprintf(stderr, "Absorbed fraction outside geometry is NOT CALCULATED.\n");
	

	//save data
	save_diff_refl(geometry, R, finished_photons, outName);

	//save number of photons to file
	FILE *numPhotFile = fopen(string(string(outName) + "_simparam.dat").c_str(), "w");
	fprintf(numPhotFile, "%d #num_photons\n", finished_photons);
	fclose(numPhotFile);

	//cleanup
	geometry_free(&geometry);
	opticalprops_free(&optProps);

	delete [] A;
	delete [] R;
	delete [] T;

	return 0;
}
