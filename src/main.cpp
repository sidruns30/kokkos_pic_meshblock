#include "global.hpp"
#include "particle_tracker.hpp"
#include "sorter_entity.hpp"
#include "sorter_buffer.hpp"
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
    const std::size_t nparticles = static_cast<int>(argc) > 1 ? std::stoi(argv[1]) : 100;
    real_t            dt         = 5.0;

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
    Kokkos::View<std::size_t[28]> tag_ctr_arr("Tag counter array");

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

    PushParticles(nparticles,
                  { 0, nx, 0, ny, 0, nz },
                  tag_ctr_arr,
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

    // PrintTags(nparticles, tag_ctr_arr, tag_arr, false);
    SortEntity( nparticles,
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
    SortBuffer( nparticles,
                tag_arr, 
                tag_ctr_arr, 
                i_arr, 
                j_arr, 
                k_arr, 
                dx_arr, 
                dy_arr, 
                dz_arr, 
                vx_arr, 
                vy_arr, 
                vz_arr);
    //PrintTags(nparticles, tag_ctr_arr, tag_arr);
  }
  std::cout << "Particle push successful" << std::endl;
  Kokkos::finalize();


  return 0;
}
