#ifndef PARTICLE_TRACKER_H
#define PARTICLE_TRACKER_H

#include "grid.hpp"
#include "util.hpp"

#include <random>

void InitializeParticleArrays(std::size_t                      nparticles,
                              const MeshBlock&                 myMeshBlock,
                              Kokkos::View<short*, CudaSpace>  tag_arr,
                              Kokkos::View<double*, CudaSpace> x_arr,
                              Kokkos::View<double*, CudaSpace> y_arr,
                              Kokkos::View<double*, CudaSpace> z_arr,
                              Kokkos::View<double*, CudaSpace> vx_arr,
                              Kokkos::View<double*, CudaSpace> vy_arr,
                              Kokkos::View<double*, CudaSpace> vz_arr);

void PushParticles(std::size_t                      nparticles,
                   Kokkos::View<double[6], Host>    MB_bounds_h,
                   Kokkos::View<short*, CudaSpace>  tag_arr,
                   Kokkos::View<size_t[28], Host>   tag_ctr_arr_h,
                   Kokkos::View<double*, CudaSpace> x_arr,
                   Kokkos::View<double*, CudaSpace> y_arr,
                   Kokkos::View<double*, CudaSpace> z_arr,
                   Kokkos::View<double*, CudaSpace> vx_arr,
                   Kokkos::View<double*, CudaSpace> vy_arr,
                   Kokkos::View<double*, CudaSpace> vz_arr,
                   double                           dt);
#endif
