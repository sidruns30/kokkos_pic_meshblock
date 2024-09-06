find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
include_directories(/Applications/miniconda3/pkgs/openmpi-4.1.1-h840aa90_1/bin)

add_library(util STATIC util.cpp)
target_include_directories(util PUBLIC ../include)
target_link_libraries(util MPI::MPI_CXX)