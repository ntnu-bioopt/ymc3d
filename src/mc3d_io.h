#ifndef MC3D_IO_H_DEFINED
#define MC3D_IO_H_DEFINED

//create geometry, optical properties from file
int opticalprops_read_from_file(opticalprops_t *optProps, char *filename);
void opticalprops_transfer_to_device(opticalprops_t *devOptProps, opticalprops_t *optProps);
int geometry_read_from_file(geometry_t *geometry, char *filename);

//mem cleanup
void opticalprops_free(opticalprops_t *optProps);
void geometry_free(geometry_t *geometry);

//general function for printing a property to binary file
void save_property(int numDims, geometry_t geometry, const double *data, const char *outName);

//drs and absorption
void save_diff_refl(geometry_t geometry, double *diffRefl, int numPhotons, char *outName);
void save_abs_map(geometry_t geometry, opticalprops_t optProps, float *abs, int numPhotons, char *outName);

#endif
