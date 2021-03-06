YMC3D
=====

YMC3D is a CUDA Monte Carlo package for tracking photons in 3D tissue inclusions, with the primary application
of generating spatially resolved diffuse reflectance as a means for investigating algorithms developed for
hyperspectral imaging. 

It has primarily been developed for internal use, but is made available for the general public with the submission
of the technical note: A. Bjorgan, M. Milanic, L. L. Randeberg, "YMC3D: GPU-accelerated 3D Monte Carlo photon tracking in tissue inclusions" (in submission). 

A description of the implementation is also available in [this technical report](http://ntnu-bioopt.github.io/publications/technical_reports/Bjorgan2015_ymc3d_implementation.pdf). 

Build instructions
------------------

Install CMake (http://www.cmake.org/cmake/resources/software.html)  

Building for Linux:
1. Navigate to build directory
2. Run cmake ..
3. Run make

Building for Windows:
1. Navigate to build directory in cmd.exe
2. Run cmake.exe ..
3. Open Visual Studio solutions files in Visual Studio
4. Build solution in Visual Studio 

Running instructions
--------------------

First, generate files containing tissue optical properties and geometry properties (see matlab/geometry/prepare_model.m for pointers to how this can be done). 

Then the simulation can be run as: 

./gpu-mc geometry.bin optprops.bid dummy_argument outputname

The dummy_argument will in the future be a text file containing definitions not connected to either tissue properties or geometry properties (e.g. the number of photons), and is mainly present as a left-over from a previous CPU version. 

Output reflectance is saved to outputname_drs.bin, and must be read using the MATLAB function read_ymc3d_outputfile.m. See also issue #2 in the issue tracker. 

License
-------

The source files for YMC3D (under src/) are released under the permissive MIT license (see src/LICENSE).

In addition, a random number generator (RNG) was used from
[GPU-MCML](https://code.google.com/p/gpumcml/) (located in gpumcml_rng/).  This
part is released under GPLv3, and thus YMC3D falls in its entirety under GPLv3
(see the LICENSE file in the root directory). If the RNG component is taken out
or replaced with a different RNG, YMC3D reverts to the MIT license. See also
issue #1 in the issue tracker.

The plan is to replace the RNG so that the project can continue under a permissive MIT license.  

safeprimes_base32.txt
---------------------

This file is obtained from [GPU-MCML](https://code.google.com/p/gpumcml/). `mc3d_rng.cu` is hard-coded to look for it in the same directory as the executable (i.e. `build/`). See src/mc3d_rng.cu.  
