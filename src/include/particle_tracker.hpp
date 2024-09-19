// HH: see comment in grid.hpp
// #include "grid.hpp"
// #include "util.hpp"
//
// #include <random>

#ifndef PARTICLE_TRACKER_H
#define PARTICLE_TRACKER_H

#include "grid.hpp"
#include "util.hpp"

// HH: see comment in util.cpp

#include <Kokkos_Core.hpp>

#include <random>

// HH: there is really no good reason for using int64_t vs std::size_t
// the second one is more standard and (importantly) portable on different
// architectures `std::size_t` is same as `size_t` but more explicit

// HH you don't really need this here
//
// extern Kokkos::View<size_t*> tag_arr;
// extern Kokkos::View<double*> x_arr, y_arr, z_arr, vx_arr, vy_arr, vz_arr;

// HH: when passing trivial types (such as integers, floats, bools, etc.), pass by value, and don't use const
// ... also Kokkos arrays need to be passed by value too
//
// void InitializeParticleArrays(const int64_t          nparticles,
// MeshBlock&             myMeshBlock,
// Kokkos::View<size_t*>& tag_arr,
// Kokkos::View<double*>& x_arr,
// Kokkos::View<double*>& y_arr,
// Kokkos::View<double*>& z_arr,
// Kokkos::View<double*>& vx_arr,
// Kokkos::View<double*>& vy_arr,
// Kokkos::View<double*>& vz_arr);
//
void InitializeParticleArrays(std::size_t           nparticles,
                              const MeshBlock&      myMeshBlock,
                              Kokkos::View<short*>  tag_arr,
                              Kokkos::View<double*> x_arr,
                              Kokkos::View<double*> y_arr,
                              Kokkos::View<double*> z_arr,
                              Kokkos::View<double*> vx_arr,
                              Kokkos::View<double*> vy_arr,
                              Kokkos::View<double*> vz_arr);

// HH: same here, and objects are usually pased by reference (or const reference)

// void PushParticles(const int64_t          nparticles,
//                    const MeshBlock        myMeshBlock,
//                    Kokkos::View<size_t*>& tag_arr,
//                    Kokkos::View<double*>& x_arr,
//                    Kokkos::View<double*>& y_arr,
//                    Kokkos::View<double*>& z_arr,
//                    Kokkos::View<double*>& vx_arr,
//                    Kokkos::View<double*>& vy_arr,
//                    Kokkos::View<double*>& vz_arr,
//                    double                 dt);
void PushParticles(std::size_t           nparticles,
                   const MeshBlock&      myMeshBlock,
                   Kokkos::View<short*>  tag_arr,
                   Kokkos::View<double*> x_arr,
                   Kokkos::View<double*> y_arr,
                   Kokkos::View<double*> z_arr,
                   Kokkos::View<double*> vx_arr,
                   Kokkos::View<double*> vy_arr,
                   Kokkos::View<double*> vz_arr,
                   double                dt);

#endif
