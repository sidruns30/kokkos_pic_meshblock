cmake_minimum_required(VERSION 3.16)
project(ParticlePusher LANGUAGES CXX)

# Set global configurations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(OpenMP_C "${CMAKE_C_COMPILER}")
set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/opt/homebrew/Cellar/libomp/17.0.2/include")
set(OpenMP_C_LIB_NAMES "libomp")
set(OpenMP_libomp_LIBRARY "omp")



find_package(MPI REQUIRED)

add_subdirectory(src)
add_subdirectory(src/grid)
add_subdirectory(src/particle_tracker)
add_subdirectory(src/utils)
