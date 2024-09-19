// HH: this `../` in include statements is typically a bad practice
//... if you want to have the headers separately, just defined their path in CMakeLists.txt

// #include "../include/grid.hpp"
// #include "../include/particle_tracker.hpp"
// #include "../include/util.hpp"

#include "grid.hpp"
#include "particle_tracker.hpp"
#include "util.hpp"

#include <Kokkos_Core.hpp>

auto main(int argc, char* argv[]) -> int {
  // Set boundaries to the grid
  float xmin  = 0.;
  float dx_mb = 1.;
  float ymin  = 0.;
  float dy_mb = 1.;
  float zmin  = 0.;
  float dz_mb = 1.;
  int   nx    = 50;
  int   ny    = 50;
  int   nz    = 50;

  const std::vector<float> xi_min_mb = { xmin, ymin, zmin };
  const std::vector<float> dxi_mb    = { dx_mb, dy_mb, dz_mb };
  const std::vector<int>   nxi_cells = { nx, ny, nz };
  auto myMeshBlock                   = MeshBlock(xi_min_mb, dxi_mb, nxi_cells);

  // Create arrays for meshblock boundaries
  std::vector<float> xmb, ymb, zmb;

  Kokkos::initialize(argc, argv);
  {
    // Initialize a few particles
    const int64_t nparticles = 100000;
    double        dt         = 0.1;

    // Initialize Particle Arrays
    Kokkos::View<short*>  tag_arr("Tag array", nparticles);
    Kokkos::View<double*> x_arr("x-position array", nparticles);
    Kokkos::View<double*> y_arr("y-position array", nparticles);
    Kokkos::View<double*> z_arr("z-position array", nparticles);
    Kokkos::View<double*> vx_arr("x-velocity array", nparticles);
    Kokkos::View<double*> vy_arr("y-velocity array", nparticles);
    Kokkos::View<double*> vz_arr("z-velocity array", nparticles);

    InitializeParticleArrays(nparticles,
                             myMeshBlock,
                             tag_arr,
                             x_arr,
                             y_arr,
                             z_arr,
                             vx_arr,
                             vy_arr,
                             vz_arr);
    PushParticles(nparticles,
                  myMeshBlock,
                  tag_arr,
                  x_arr,
                  y_arr,
                  z_arr,
                  vx_arr,
                  vy_arr,
                  vz_arr,
                  dt);
  }
  Kokkos::finalize();

  // Make Kokkos arrays for particles
  /*
    Question: view is created on the cpu. Would accessing it on the gpu be a lot slower?
    HH: yes, in fact infinitely slower, since you will not be able to access it at all.
  */

  return 0;
}
