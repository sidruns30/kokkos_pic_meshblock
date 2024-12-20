#include "global.hpp"
#include "particle_tracker.hpp"
#include "sorter_buffer.hpp"
#include "sorter_entity.hpp"
#include "sorter_thrust.hpp"
#include "timer.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>

auto main(int argc, char* argv[]) -> int {
  // Set boundaries to the grid
  int nx = 50;
  int ny = 70;
  int nz = 63;

  Kokkos::initialize(argc, argv);
  {
    const std::size_t nparticles = argc > 1 ? std::stoi(argv[1]) : 100;

    real_t dt = 5.0;

    std::cout << "Number of particles: " << nparticles << std::endl;
    // Initialize Particle Arrays
    Kokkos::View<short*>          tag_arr("Tag array", nparticles);
    Kokkos::View<int*>            i_arr("i-position array", nparticles);
    Kokkos::View<int*>            j_arr("j-position array", nparticles);
    Kokkos::View<int*>            k_arr("k-position array", nparticles);
    Kokkos::View<float*>          dx_arr("dx-position array", nparticles);
    Kokkos::View<float*>          dy_arr("dy-position array", nparticles);
    Kokkos::View<float*>          dz_arr("dz-position array", nparticles);
    Kokkos::View<real_t*>         vx_arr("x-velocity array", nparticles);
    Kokkos::View<real_t*>         vy_arr("y-velocity array", nparticles);
    Kokkos::View<real_t*>         vz_arr("z-velocity array", nparticles);
    Kokkos::View<std::size_t[29]> tag_ctr_cumsum("Tag counter array");

    InitializeParticleArrays(nparticles,
                             { 0, nx, 0, ny, 0, nz },
                             tag_arr,
                             i_arr,
                             j_arr,
                             k_arr,
                             dx_arr,
                             dy_arr,
                             dz_arr,
                             vx_arr,
                             vy_arr,
                             vz_arr);

    Kokkos::View<short*>          tag_arr_bckp("Tag array", nparticles);
    Kokkos::View<int*>            i_arr_bckp("i-position array", nparticles);
    Kokkos::View<int*>            j_arr_bckp("j-position array", nparticles);
    Kokkos::View<int*>            k_arr_bckp("k-position array", nparticles);
    Kokkos::View<float*>          dx_arr_bckp("dx-position array", nparticles);
    Kokkos::View<float*>          dy_arr_bckp("dy-position array", nparticles);
    Kokkos::View<float*>          dz_arr_bckp("dz-position array", nparticles);
    Kokkos::View<real_t*>         vx_arr_bckp("x-velocity array", nparticles);
    Kokkos::View<real_t*>         vy_arr_bckp("y-velocity array", nparticles);
    Kokkos::View<real_t*>         vz_arr_bckp("z-velocity array", nparticles);
    Kokkos::View<std::size_t[29]> tag_ctr_cumsum_bckp("Tag counter array");

    PushParticles(nparticles,
                  { 0, nx, 0, ny, 0, nz },
                  tag_ctr_cumsum,
                  tag_arr,
                  i_arr,
                  j_arr,
                  k_arr,
                  dx_arr,
                  dy_arr,
                  dz_arr,
                  vx_arr,
                  vy_arr,
                  vz_arr,
                  dt);

    Kokkos::deep_copy(tag_arr_bckp, tag_arr);
    Kokkos::deep_copy(i_arr_bckp, i_arr);
    Kokkos::deep_copy(j_arr_bckp, j_arr);
    Kokkos::deep_copy(k_arr_bckp, k_arr);
    Kokkos::deep_copy(dx_arr_bckp, dx_arr);
    Kokkos::deep_copy(dy_arr_bckp, dy_arr);
    Kokkos::deep_copy(dz_arr_bckp, dz_arr);
    Kokkos::deep_copy(vx_arr_bckp, vx_arr);
    Kokkos::deep_copy(vy_arr_bckp, vy_arr);
    Kokkos::deep_copy(vz_arr_bckp, vz_arr);
    Kokkos::deep_copy(tag_ctr_cumsum_bckp, tag_ctr_cumsum);

    // for (int iteration = 0; iteration < 20; iteration++) {
    //   SortThrust(nparticles,
    //              tag_arr,
    //              i_arr,
    //              j_arr,
    //              k_arr,
    //              dx_arr,
    //              dy_arr,
    //              dz_arr,
    //              vx_arr,
    //              vy_arr,
    //              vz_arr);
    //   Kokkos::deep_copy(tag_arr, tag_arr_bckp);
    //   Kokkos::deep_copy(i_arr, i_arr_bckp);
    //   Kokkos::deep_copy(j_arr, j_arr_bckp);
    //   Kokkos::deep_copy(k_arr, k_arr_bckp);
    //   Kokkos::deep_copy(dx_arr, dx_arr_bckp);
    //   Kokkos::deep_copy(dy_arr, dy_arr_bckp);
    //   Kokkos::deep_copy(dz_arr, dz_arr_bckp);
    //   Kokkos::deep_copy(vx_arr, vx_arr_bckp);
    //   Kokkos::deep_copy(vy_arr, vy_arr_bckp);
    //   Kokkos::deep_copy(vz_arr, vz_arr_bckp);
    //   Kokkos::deep_copy(tag_ctr_cumsum, tag_ctr_cumsum_bckp);
    // }
    for (int iteration = 0; iteration < 20; iteration++) {
      SortBuffer(nparticles,
                 tag_arr,
                 tag_ctr_cumsum,
                 i_arr,
                 j_arr,
                 k_arr,
                 dx_arr,
                 dy_arr,
                 dz_arr,
                 vx_arr,
                 vy_arr,
                 vz_arr);
      Kokkos::deep_copy(tag_arr, tag_arr_bckp);
      Kokkos::deep_copy(i_arr, i_arr_bckp);
      Kokkos::deep_copy(j_arr, j_arr_bckp);
      Kokkos::deep_copy(k_arr, k_arr_bckp);
      Kokkos::deep_copy(dx_arr, dx_arr_bckp);
      Kokkos::deep_copy(dy_arr, dy_arr_bckp);
      Kokkos::deep_copy(dz_arr, dz_arr_bckp);
      Kokkos::deep_copy(vx_arr, vx_arr_bckp);
      Kokkos::deep_copy(vy_arr, vy_arr_bckp);
      Kokkos::deep_copy(vz_arr, vz_arr_bckp);
      Kokkos::deep_copy(tag_ctr_cumsum, tag_ctr_cumsum_bckp);
    }
    for (int iteration = 0; iteration < 20; iteration++) {
      SortEntity(nparticles,
                 tag_arr,
                 i_arr,
                 j_arr,
                 k_arr,
                 dx_arr,
                 dy_arr,
                 dz_arr,
                 vx_arr,
                 vy_arr,
                 vz_arr);
      Kokkos::deep_copy(tag_arr, tag_arr_bckp);
      Kokkos::deep_copy(i_arr, i_arr_bckp);
      Kokkos::deep_copy(j_arr, j_arr_bckp);
      Kokkos::deep_copy(k_arr, k_arr_bckp);
      Kokkos::deep_copy(dx_arr, dx_arr_bckp);
      Kokkos::deep_copy(dy_arr, dy_arr_bckp);
      Kokkos::deep_copy(dz_arr, dz_arr_bckp);
      Kokkos::deep_copy(vx_arr, vx_arr_bckp);
      Kokkos::deep_copy(vy_arr, vy_arr_bckp);
      Kokkos::deep_copy(vz_arr, vz_arr_bckp);
      Kokkos::deep_copy(tag_ctr_cumsum, tag_ctr_cumsum_bckp);
    }
  }
  Kokkos::finalize();

  return 0;
}
