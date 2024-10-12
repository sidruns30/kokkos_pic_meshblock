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
  Kokkos::View<size_t[28], CudaSpace> tag_ctr_arr;
  Kokkos::View<double[6],  CudaSpace> MB_bounds;
  Kokkos::deep_copy(MB_bounds, MB_bounds_h);

  // SS: A bit suspicious if this enum will be captured by the Kokkos::lambda...
  enum IndexPosition {
                      XYZ          = 0,
                      YZXmin       = 1,
                      YZXmax       = 2,
                      XZYmin       = 3,
                      XZYmax       = 4,
                      XYZmin       = 5,
                      XYZmax       = 6,
                      ZXminYmin    = 7,
                      ZXminYmax    = 8,
                      ZXmaxYmin    = 9,
                      ZXmaxYmax    = 10,
                      YXminZmin    = 11,
                      YXminZmax    = 12,
                      YXmaxZmin    = 13,
                      YXmaxZmax    = 14,
                      XYminZmin    = 15,
                      XYminZmax    = 16,
                      XYmaxZmin    = 17,
                      XYmaxZmax    = 18,
                      XminYminZmin = 19,
                      XminYminZmax = 20,
                      XminYmaxZmin = 21,
                      XminYmaxZmax = 22,
                      XmaxYminZmin = 23,
                      XmaxYminZmax = 24,
                      XmaxYmaxZmin = 25,
                      XmaxYmaxZmax = 26,
                      BDRY_OUT     = 27};

  Kokkos::parallel_for(
    "Particle pusher loop",
    Kokkos::RangePolicy<Kokkos::Cuda>(0, nparticles),
    KOKKOS_LAMBDA(const std::size_t p) {
      x_arr(p) += vx_arr(p) * dt;
      y_arr(p) += vy_arr(p) * dt;
      z_arr(p) += vz_arr(p) * dt;

      double x = x_arr(p);
      double y = y_arr(p);
      double z = z_arr(p);

      double xmin = MB_bounds(0);
      double xmax = MB_bounds(1);
      double ymin = MB_bounds(2);
      double ymax = MB_bounds(3);
      double zmin = MB_bounds(4);
      double zmax = MB_bounds(5);

      /* SS:  The next long bit is to compute the tag
              I am currently not sure how I can take a KOKKOS_INLINE_FUNCTION
              e.g., `ComputeTag' from outside and put it inside this lambda.
              Therefore, I am performing the compute tag operation explicitly
              inside this lambda.
              Additionally, I don't know how to define the MeshBlock class on
              the device. So I pass the MB_bounds Kokkos view. 
      */
      if (z > zmin and z < zmax) {
      // y-z contained
      if (y > ymin and y < ymax) {
        // x, y, z contained
        if (x > xmin and x < xmax) {
            //x_arr(p) = XYZ;
            //tag_ctr_arr(tag_arr(p)) += 1;
            return;
        }
        // yz contained, x < xmin
        else if (x < xmin) {
          //tag_arr(p) = YZXmin;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        }
        // yz contained, x > xmax
        else {
          //tag_arr(p) = YZXmax;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        }
      }
      // x-z contained
      else if (x > xmin and x < xmax) {
        // xz contained, y < min
        if (y < ymin) {
          //tag_arr(p) = XZYmin;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        }
        // xz contained, y > ymax
        else {
          //tag_arr(p) = XZYmax;
          //tag_ctr_arr(tag_arr(p)) += 1;
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
          //tag_arr(p) = XYZmin;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        } else {
          //tag_arr(p) = XYZmax;
          //tag_ctr_arr(tag_arr(p)) += 1;
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
          //tag_arr(p) = ZXminYmin;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        } else {
          //tag_arr(p) = ZXminYmax;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        }
      } else {
        if (y < ymin) {
          //tag_arr(p) = ZXmaxYmin;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        } else {
          //tag_arr(p) = ZXmaxYmax;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        }
      }
    }
    // y contained only
    else if (y > ymin and y < ymax) {
      if (x < xmin) {
        if (z < zmin) {
          //tag_arr(p) = YXminZmin;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        } else {
          //tag_arr(p) = YXminZmax;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        }
      } else {
        if (z < zmin) {
          //tag_arr(p) = YXmaxZmin;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        } else {
          //tag_arr(p) = YXmaxZmax;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        }
      }
    }
    // x contained only
    else if (x > xmin and x < xmax) {
      if (y < ymin) {
        if (z < zmin) {
          //tag_arr(p) = XYminZmin;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        } else {
          //tag_arr(p) = XYminZmax;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        }
      } else {
        if (z < zmin) {
          //tag_arr(p) = XYmaxZmin;
          //tag_ctr_arr(tag_arr(p)) += 1;
          return;
        } else {
          //tag_arr(p) = XYmaxZmax;
          //tag_ctr_arr(tag_arr(p)) += 1;
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
            //tag_arr(p) = XminYminZmin;
            //tag_ctr_arr(tag_arr(p)) += 1;
            return;
          } else {
            //tag_arr(p) = XminYminZmax;
            //tag_ctr_arr(tag_arr(p)) += 1;
            return;
          }
        } else {
          if (z < zmin) {
            //tag_arr(p) = XminYmaxZmin;
            //tag_ctr_arr(tag_arr(p)) += 1;
            return;
          } else {
            //tag_arr(p) = XminYmaxZmax;
            //tag_ctr_arr(tag_arr(p)) += 1;
            return;
          }
        }
      } else {
        if (y < ymin) {
          if (z < zmin) {
            //tag_arr(p) = XmaxYminZmin;
            //tag_ctr_arr(tag_arr(p)) += 1;
            return;
          } else {
            //tag_arr(p) = XmaxYminZmax;
            //tag_ctr_arr(tag_arr(p)) += 1;
            return;
          }
        } else {
          if (z < zmin) {
            //tag_arr(p) = XmaxYmaxZmin;
            //tag_ctr_arr(tag_arr(p)) += 1;
            return;
          } else {
            //tag_arr(p) = XmaxYmaxZmax;
            //tag_ctr_arr(tag_arr(p)) += 1;
            return;
          }
        }
      }
    }
    
    });

  // End of Kokkos::Lambda and parallel for
  Kokkos::deep_copy(tag_ctr_arr_h, tag_ctr_arr);

}

// 