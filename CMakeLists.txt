cmake_minimum_required(VERSION 3.16)
project(ParticlePusher LANGUAGES CXX)

# Set global configurations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Kokkos REQUIRED HINTS /home/siddhant/kokkos-install)

add_subdirectory(src)
