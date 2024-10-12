#ifndef UTIL_H
#define UTIL_H
#include <vector>
#include <Kokkos_Core.hpp>

void PrintVec(std::vector<float> vec);
typedef Kokkos::CudaSpace CudaSpace;
typedef Kokkos::HostSpace Host;
#endif
