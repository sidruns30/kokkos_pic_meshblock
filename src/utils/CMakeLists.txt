include(FetchContent)
FetchContent_Declare(
  Kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG master
)

FetchContent_MakeAvailable(Kokkos)


find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
include_directories(/Applications/miniconda3/pkgs/openmpi-4.1.1-h840aa90_1/bin)

add_library(util STATIC util.cpp)
target_include_directories(util PUBLIC ../include)
target_link_libraries(util PUBLIC MPI::MPI_CXX)
target_link_libraries(util PUBLIC Kokkos::kokkos)
