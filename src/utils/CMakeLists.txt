# HH: usually, all external libraries are loaded in the root CMakeLists.txt

# include(FetchContent)
# FetchContent_Declare(
#   Kokkos
#   GIT_REPOSITORY https://github.com/kokkos/kokkos.git
#   GIT_TAG master
# )
#
# FetchContent_MakeAvailable(Kokkos)
#
#
# find_package(MPI REQUIRED)
# include_directories(${MPI_INCLUDE_PATH})
# include_directories(/Applications/miniconda3/pkgs/openmpi-4.1.1-h840aa90_1/bin)

add_library(util util.cpp)

# HH: just link Kokkos globally
#
# target_link_libraries(util PUBLIC Kokkos::kokkos)

# HH: you don't need to include one-by-one for each library. (see root CMakeLists.txt)

# target_include_directories(util PUBLIC ../include)
# target_link_libraries(util PUBLIC MPI::MPI_CXX)
