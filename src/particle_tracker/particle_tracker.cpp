#include "particle_tracker.hpp"

// Initialize the particle arrays
// Initialization is done on the host (CPU)
void InitializeParticleArrays(std::size_t           nparticles,
                              const MeshBlock&      myMeshBlock,
                              Kokkos::View<short*,  CudaSpace>  tag_arr,
                              Kokkos::View<double*, CudaSpace>  x_arr,
                              Kokkos::View<double*, CudaSpace>  y_arr,
                              Kokkos::View<double*, CudaSpace>  z_arr,
                              Kokkos::View<double*, CudaSpace>  vx_arr,
                              Kokkos::View<double*, CudaSpace>  vy_arr,
                              Kokkos::View<double*, CudaSpace>  vz_arr) {
  // Create Random Distributions to Initialize Particle Arrays
  std::uniform_real_distribution<> dis_x(myMeshBlock.xmin, myMeshBlock.xmax);
  std::uniform_real_distribution<> dis_y(myMeshBlock.ymin, myMeshBlock.ymax);
  std::uniform_real_distribution<> dis_z(myMeshBlock.zmin, myMeshBlock.zmax);
  std::uniform_real_distribution<> dis_vel(-1, 1);
  std::random_device               rd;
  std::mt19937                     gen(rd());
  // mirror view
  auto x_arr_h   = Kokkos::create_mirror_view(x_arr);
  auto y_arr_h   = Kokkos::create_mirror_view(y_arr);
  auto z_arr_h   = Kokkos::create_mirror_view(z_arr);
  auto vx_arr_h  = Kokkos::create_mirror_view(vx_arr);
  auto vy_arr_h  = Kokkos::create_mirror_view(vy_arr);
  auto vz_arr_h  = Kokkos::create_mirror_view(vz_arr);
  auto tag_arr_h = Kokkos::create_mirror_view(tag_arr);
  
  for (auto p = 0; p < nparticles; p++) {
    x_arr_h(p)   = dis_x(gen);
    y_arr_h(p)   = dis_y(gen);
    z_arr_h(p)   = dis_z(gen);
    vx_arr_h(p)  = dis_vel(gen);
    vy_arr_h(p)  = dis_vel(gen);
    vz_arr_h(p)  = dis_vel(gen);
  }

  Kokkos::deep_copy(x_arr, x_arr_h);
  Kokkos::deep_copy(y_arr, y_arr_h);
  Kokkos::deep_copy(z_arr, z_arr_h);
  Kokkos::deep_copy(vx_arr, vx_arr_h);
  Kokkos::deep_copy(vy_arr, vy_arr_h);
  Kokkos::deep_copy(vz_arr, vz_arr_h);
  Kokkos::deep_copy(tag_arr, tag_arr_h);
}

// Particles are pushed on the device (GPU)
void PushParticles( std::size_t           nparticles,
                    Kokkos::View<double[6],     Host>  MB_bounds_h,
                    Kokkos::View<short*,      CudaSpace>  tag_arr,
                    Kokkos::View<size_t[28],  Host>  tag_ctr_arr_h,
                    Kokkos::View<double*,     CudaSpace>  x_arr,
                    Kokkos::View<double*,     CudaSpace>  y_arr,
                    Kokkos::View<double*,     CudaSpace>  z_arr,
                    Kokkos::View<double*,     CudaSpace>  vx_arr,
                    Kokkos::View<double*,     CudaSpace>  vy_arr,
                    Kokkos::View<double*,     CudaSpace>  vz_arr,
                    double                    dt) {

  // Array that keeps track of the number of particles sent to each neighbor
  Kokkos::View<size_t[28], CudaSpace> tag_ctr_arr("tag counter array");
  Kokkos::View<double[6],  CudaSpace> MB_bounds("Meshblock bounds array");
  Kokkos::deep_copy(MB_bounds, MB_bounds_h);

  // SS: Changed enum to a bunch of size_t's so hopefully everything is sent
  //     to the GPU
  const size_t XYZ             = 0;
  const size_t YZXmin          = 1;
  const size_t YZXmax          = 2;
  const size_t XZYmin          = 3;
  const size_t XZYmax          = 4;
  const size_t XYZmin          = 5;
  const size_t XYZmax          = 6;
  const size_t ZXminYmin       = 7;
  const size_t ZXminYmax       = 8;
  const size_t ZXmaxYmin       = 9;
  const size_t ZXmaxYmax       = 10;
  const size_t YXminZmin       = 11;
  const size_t YXminZmax       = 12;
  const size_t YXmaxZmin       = 13;
  const size_t YXmaxZmax       = 14;
  const size_t XYminZmin       = 15;
  const size_t XYminZmax       = 16;
  const size_t XYmaxZmin       = 17;
  const size_t XYmaxZmax       = 18;
  const size_t XminYminZmin    = 19;
  const size_t XminYminZmax    = 20;
  const size_t XminYmaxZmin    = 21;
  const size_t XminYmaxZmax    = 22;
  const size_t XmaxYminZmin    = 23;
  const size_t XmaxYminZmax    = 24;
  const size_t XmaxYmaxZmin    = 25;
  const size_t XmaxYmaxZmax    = 26;
  const size_t BDRY_OUT        = 27;
  
  const double xmin = MB_bounds_h(0);
  const double xmax = MB_bounds_h(1);
  const double ymin = MB_bounds_h(2);
  const double ymax = MB_bounds_h(3);
  const double zmin = MB_bounds_h(4);
  const double zmax = MB_bounds_h(5);


  /* SS:  The next long bit is to compute the tag
          I am currently not sure how I can take a KOKKOS_INLINE_FUNCTION
          e.g., `ComputeTag' from outside and put it inside this lambda.
          Therefore, I am performing the compute tag operation explicitly
          inside this lambda.
          Additionally, I don't know how to define the MeshBlock class on
          the device. So I pass the MB_bounds Kokkos view. 
  */

  Kokkos::parallel_for(
    "Particle pusher loop",
    Kokkos::RangePolicy<Kokkos::Cuda>(0, nparticles),
    KOKKOS_LAMBDA(const std::size_t p) {
      x_arr(p) += vx_arr(p) * dt;
      y_arr(p) += vy_arr(p) * dt;
      z_arr(p) += vz_arr(p) * dt;

      const double x = x_arr(p);
      const double y = y_arr(p);
      const double z = z_arr(p);

      
      if (z > zmin and z < zmax) {
      // y-z contained
      if (y > ymin and y < ymax) {
        // x, y, z contained
        if (x > xmin and x < xmax) {
            tag_arr(p) = XYZ;
            Kokkos::atomic_increment(&tag_ctr_arr(XYZ));
            return;
        }
        // yz contained, x < xmin
        else if (x < xmin) {
          tag_arr(p) = YZXmin;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
        // yz contained, x > xmax
        else {
          tag_arr(p) = YZXmax;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
      }
      // x-z contained
      else if (x > xmin and x < xmax) {
        // xz contained, y < min
        if (y < ymin) {
          tag_arr(p) = XZYmin;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
        // xz contained, y > ymax
        else {
          tag_arr(p) = XZYmax;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
      }
    }
    // x contained, z not conatined
    else if (x > xmin and x < xmax) {
      // x-y contained
      if (y > ymin and y < ymax) {
        // x-y contained, z < zmin
        if (z < zmin) {
          tag_arr(p) = XYZmin;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        } else {
          tag_arr(p) = XYZmax;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
      }
    }
    // The particle now lies at least in the edge or the corner MeshBlock
    // Let us do the edges first
    // We know that the particle does not have x or z coordinate in the block
    // z contained only
    if (z > zmin and z < zmax) {
      if (x < xmin) {
        if (y < ymin) {
          tag_arr(p) = ZXminYmin;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        } else {
          tag_arr(p) = ZXminYmax;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
      } else {
        if (y < ymin) {
          tag_arr(p) = ZXmaxYmin;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        } else {
          tag_arr(p) = ZXmaxYmax;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
      }
    }
    // y contained only
    else if (y > ymin and y < ymax) {
      if (x < xmin) {
        if (z < zmin) {
          tag_arr(p) = YXminZmin;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        } else {
          tag_arr(p) = YXminZmax;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
      } else {
        if (z < zmin) {
          tag_arr(p) = YXmaxZmin;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        } else {
          tag_arr(p) = YXmaxZmax;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
      }
    }
    // x contained only
    else if (x > xmin and x < xmax) {
      if (y < ymin) {
        if (z < zmin) {
          tag_arr(p) = XYminZmin;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        } else {
          tag_arr(p) = XYminZmax;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
      } else {
        if (z < zmin) {
          tag_arr(p) = XYmaxZmin;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        } else {
          tag_arr(p) = XYmaxZmax;
          Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
          return;
        }
      }
    }
    //  In the cases below, the particle contains none of the coordinates
    //  so it must be in one of the corner cells
    else {
      if (x < xmin) {
        if (y < ymin) {
          if (z < zmin) {
            tag_arr(p) = XminYminZmin;
            Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
            return;
          } else {
            tag_arr(p) = XminYminZmax;
            Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
            return;
          }
        } else {
          if (z < zmin) {
            tag_arr(p) = XminYmaxZmin;
            Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
            return;
          } else {
            tag_arr(p) = XminYmaxZmax;
            Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
            return;
          }
        }
      } else {
        if (y < ymin) {
          if (z < zmin) {
            tag_arr(p) = XmaxYminZmin;
            Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
            return;
          } else {
            tag_arr(p) = XmaxYminZmax;
            Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
            return;
          }
        } else {
          if (z < zmin) {
            tag_arr(p) = XmaxYmaxZmin;
            Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
            return;
          } else {
            tag_arr(p) = XmaxYmaxZmax;
            Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
            return;
          }
        }
      }
    }
    
    });

  // End of Kokkos::Lambda and parallel for
  //Kokkos::deep_copy(tag_ctr_arr_h, tag_ctr_arr);

}