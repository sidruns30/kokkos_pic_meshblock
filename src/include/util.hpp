// HH: this is the opposite case of what I commented in util.cpp.
// ... you're not really using <random>, or <algorithm> or any of the mentioned headers
// ... at all here. so just don't define them here.
//
// HH: also, always put the include guards(!!!)

#ifndef UTIL_H
#define UTIL_H

#include <vector>
// #include <mpi.h>
//  #include <iostream>
//  #include <random>
//  #include <algorithm>
//  #include <cassert>
//  #include <Kokkos_Core.hpp>

void PrintVec(std::vector<float> vec);

#endif
