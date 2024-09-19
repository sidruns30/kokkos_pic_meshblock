cmake_minimum_required(VERSION 3.16)
project(ParticlePusher LANGUAGES CXX)

# Set global configurations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# HH: I think this is not needed
#
# set(OpenMP_C "${CMAKE_C_COMPILER}")
# set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/opt/homebrew/Cellar/libomp/17.0.2/include")
# set(OpenMP_C_LIB_NAMES "libomp")
# set(OpenMP_libomp_LIBRARY "omp")

# let's comment this out for now
# find_package(MPI REQUIRED)

# HH: see my comment in src/utils/CMakeLists.txt

include(FetchContent)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG master
)
FetchContent_MakeAvailable(Kokkos)

# HH: include the headers for all libraries here and link to kokkos
#
include_directories(${CMAKE_SOURCE_DIR}/src/include)
link_libraries(Kokkos::kokkos)

add_subdirectory(src)

# HH: it's a better practice to include subdirectories in src/CMakelists.txt itself
#
# add_subdirectory(src/grid)
# add_subdirectory(src/particle_tracker)
# add_subdirectory(src/utils)
#

