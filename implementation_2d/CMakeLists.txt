cmake_minimum_required(VERSION 3.18)
project(cg_poisson LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(CG_NATIVE "Tune host code for the build machine (-march=native)" ON)
option(CG_ENABLE_CUDA "Build the MPI+CUDA solver if a CUDA compiler is found" ON)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

# Flags shared by every host executable.
add_library(cg_common INTERFACE)
target_include_directories(cg_common INTERFACE ${PROJECT_SOURCE_DIR}/src)
target_compile_options(cg_common INTERFACE -Wall -Wextra)
if(CG_NATIVE)
  target_compile_options(cg_common INTERFACE -march=native)
endif()

# Serial baseline: OpenMP SIMD hints only, no threads, no MPI.
add_executable(cg_serial src/cg_serial.cpp)
target_link_libraries(cg_serial PRIVATE cg_common)
target_compile_options(cg_serial PRIVATE -fopenmp-simd)

add_executable(cg_mpi_omp src/cg_mpi_omp.cpp)
target_link_libraries(cg_mpi_omp PRIVATE cg_common MPI::MPI_CXX OpenMP::OpenMP_CXX)

if(CG_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # PTX for compute capability 7.0 runs on every newer GPU via the driver's
    # JIT; pass e.g. -DCMAKE_CUDA_ARCHITECTURES=80 for native A100 code.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 70-virtual)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(cg_cuda_kernels STATIC src/cuda_kernels.cu)
    set_target_properties(cg_cuda_kernels PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON)
    target_include_directories(cg_cuda_kernels PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(cg_cuda_kernels PUBLIC CUDA::cudart)

    # OpenMP only speeds up the host-side setup and final error check here.
    add_executable(cg_mpi_cuda src/cg_mpi_cuda.cpp)
    target_link_libraries(cg_mpi_cuda PRIVATE cg_common cg_cuda_kernels MPI::MPI_CXX OpenMP::OpenMP_CXX)
  else()
    message(STATUS "No CUDA compiler found: cg_mpi_cuda will not be built")
  endif()
endif()
