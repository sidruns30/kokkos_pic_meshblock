#include <random>
#include "util.hpp"
#include "grid.hpp"

#ifndef PARTICLE_TRACKER_H
    #define PARTICLE_TRACKER_H
    extern Kokkos::View <size_t*> tag_arr;
    extern Kokkos::View <double*> x_arr, y_arr, z_arr, 
                                vx_arr, vy_arr, vz_arr;

    void InitializeParticleArrays(const int64_t nparticles, MeshBlock &myMeshBlock,
                              Kokkos::View <size_t*> &tag_arr, Kokkos::View <double*> &x_arr,
                              Kokkos::View <double*> &y_arr, Kokkos::View <double*> &z_arr ,
                              Kokkos::View <double*> &vx_arr, Kokkos::View <double*> &vy_arr,
                              Kokkos::View <double*> &vz_arr);

    void PushParticles(const int64_t nparticles, const MeshBlock myMeshBlock,
                              Kokkos::View <size_t*> &tag_arr, Kokkos::View <double*> &x_arr,
                              Kokkos::View <double*> &y_arr, Kokkos::View <double*> &z_arr ,
                              Kokkos::View <double*> &vx_arr, Kokkos::View <double*> &vy_arr,
                              Kokkos::View <double*> &vz_arr, double dt);
#endif