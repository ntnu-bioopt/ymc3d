cmake_minimum_required(VERSION 2.6)
project(ymc3d)
set(CMAKE_BUILD_TYPE Release)

INCLUDE_DIRECTORIES(src/ gpumcml_rng/)


FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(CUDA)

list (APPEND CUDA_NVCC_FLAGS --compiler-bindir /usr/bin/gcc-4.7)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_30 -Xptxas -v -use_fast_math)
cuda_add_executable(gpu-mc src/mc3d_io.cu src/mc3d_photons.cu src/mc3d_main.cu src/mc3d_gpu.cu)
