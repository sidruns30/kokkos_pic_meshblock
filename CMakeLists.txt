cmake_minimum_required(VERSION 3.16)
project(CPP-Kokkos-Boilerplate LANGUAGES CXX)

# Set global configurations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG master
)

FetchContent_MakeAvailable(Kokkos)

find_package(MPI REQUIRED)

add_subdirectory(src)
add_subdirectory(src/grid)
add_subdirectory(src/particle_tracker)
add_subdirectory(src/utils)
