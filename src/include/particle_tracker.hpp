#ifndef PARTICLE_TRACKER_H
#define PARTICLE_TRACKER_H

#include "global.hpp"

#include <Kokkos_Core.hpp>

#include <array>

void InitializeParticleArrays(std::size_t,
                              const std::array<int, 6>&,
                              Kokkos::View<short*>,
                              Kokkos::View<int*>,
                              Kokkos::View<int*>,
                              Kokkos::View<int*>,
                              Kokkos::View<float*>,
                              Kokkos::View<float*>,
                              Kokkos::View<float*>,
                              Kokkos::View<real_t*>,
                              Kokkos::View<real_t*>,
                              Kokkos::View<real_t*>);

void PushParticles(std::size_t,
                   const std::array<int, 6>&,
                   Kokkos::View<std::size_t[29]>,
                   Kokkos::View<short*>,
                   Kokkos::View<int*>,
                   Kokkos::View<int*>,
                   Kokkos::View<int*>,
                   Kokkos::View<float*>,
                   Kokkos::View<float*>,
                   Kokkos::View<float*>,
                   Kokkos::View<real_t*>,
                   Kokkos::View<real_t*>,
                   Kokkos::View<real_t*>,
                   double);

void PrintTags(std::size_t,
               Kokkos::View<std::size_t[28]>,
               Kokkos::View<short*>,
               bool = true);

#endif
