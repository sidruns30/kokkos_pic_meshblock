#ifndef UTIL_H
#define UTIL_H

#include <Kokkos_Core.hpp>

#include <vector>

void                      PrintVec(std::vector<float> vec);
typedef Kokkos::CudaSpace CudaSpace;
typedef Kokkos::HostSpace Host;

#endif
