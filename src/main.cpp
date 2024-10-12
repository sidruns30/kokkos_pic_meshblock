#include "grid.hpp"
#include "particle_tracker.hpp"
#include "util.hpp"

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
    // SS: Just being a bit more explicit about what memory space the Views are initialized
    Kokkos::View<short*,      CudaSpace>  tag_arr("Tag array", nparticles);
    Kokkos::View<double*,     CudaSpace>  x_arr("x-position array", nparticles);
    Kokkos::View<double*,     CudaSpace>  y_arr("y-position array", nparticles);
    Kokkos::View<double*,     CudaSpace>  z_arr("z-position array", nparticles);
    Kokkos::View<double*,     CudaSpace>  vx_arr("x-velocity array", nparticles);
    Kokkos::View<double*,     CudaSpace>  vy_arr("y-velocity array", nparticles);
    Kokkos::View<double*,     CudaSpace>  vz_arr("z-velocity array", nparticles);
    Kokkos::View<double[6],   Host>  MB_bounds_h("Bounds of the MeshBlock");
    Kokkos::View<size_t[28],  Host>  tag_ctr_arr_h("Tag counter array");

    // Populate the MB bounds array on Host
    MB_bounds_h(0) = myMeshBlock.xmin;
    MB_bounds_h(1) = myMeshBlock.xmax;
    MB_bounds_h(2) = myMeshBlock.ymin;
    MB_bounds_h(3) = myMeshBlock.ymax;
    MB_bounds_h(4) = myMeshBlock.zmin;
    MB_bounds_h(5) = myMeshBlock.zmax;


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
                  MB_bounds_h,
                  tag_arr,
                  tag_ctr_arr_h,
                  x_arr,
                  y_arr,
                  z_arr,
                  vx_arr,
                  vy_arr,
                  vz_arr,
                  dt);
    
  }
  std::cout << "Particle push successful" << std::endl;
  Kokkos::finalize();

  return 0;
}
