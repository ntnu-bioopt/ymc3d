//create geometry, optical properties from file
int createOptProps(OpticalProps *optProps, char *filename);
void createDeviceOptProps(OpticalProps *devOptProps, OpticalProps *optProps);
int createGeometry(Geometry *geometry, char *filename);

//seeds for random number generator
void createRNGSeeds(RNGSeeds *rngSeeds, UINT64 seed, int num_photons_per_packet);


//mem cleanup
void freeOptProps(OpticalProps *optProps);
void freeRNGSeeds(RNGSeeds *rngSeeds);
void freeGeometry(Geometry *geometry);

//general function for printing a property to binary file
void saveProperty(int numDims, Geometry geometry, const double *data, const char *outName);

//drs and absorption
void saveDiffRefl(Geometry geometry, double *diffRefl, int numPhotons, char *outName);
void saveBeam(Geometry geometry, double *beam, char *outName);
void saveAbsMap(Geometry geometry, OpticalProps optProps, float *abs, int numPhotons, char *outName);
