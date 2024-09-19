#include "../include/particle_tracker.hpp"

// Initialize the particle arrays
void InitializeParticleArrays(const int64_t nparticles, MeshBlock &myMeshBlock,
                              Kokkos::View <size_t*> &tag_arr, Kokkos::View <double*> &x_arr,
                              Kokkos::View <double*> &y_arr, Kokkos::View <double*> &z_arr ,
                              Kokkos::View <double*> &vx_arr, Kokkos::View <double*> &vy_arr,
                              Kokkos::View <double*> &vz_arr)
{
    // Create Random Distributions to Initialize Particle Arrays
    std::uniform_real_distribution<> dis_x(myMeshBlock.xmin, myMeshBlock.xmax);
    std::uniform_real_distribution<> dis_y(myMeshBlock.ymin, myMeshBlock.ymax);
    std::uniform_real_distribution<> dis_z(myMeshBlock.zmin, myMeshBlock.zmax);
    std::uniform_real_distribution<> dis_vel(-1, 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    // mirror view
    // Kokkos for rng

    auto x_arr_h = Kokkos::create_mirror_view(x_arr);
    // ... init x_arr_h on CPU


    for (int64_t index=0; index<nparticles; index++)
    {
      x_arr_h(index) = dis_x(gen);
      y_arr(index) = dis_y(gen);
      z_arr(index) = dis_z(gen);
      vx_arr(index) = dis_vel(gen);
      vy_arr(index) = dis_vel(gen);
      vz_arr(index) = dis_vel(gen);
      tag_arr(index) = myMeshBlock.ComputeTag(x_arr(index), y_arr(index), z_arr(index));
  }
  Kokkos::deep_copy(x_arr, x_arr_h);
}


void PushParticles(const int64_t nparticles, const MeshBlock myMeshBlock,
                    Kokkos::View <size_t*> tag_arr, Kokkos::View <double*> x_arr,
                    Kokkos::View <double*> y_arr, Kokkos::View <double*> z_arr ,
                    Kokkos::View <double*> vx_arr, Kokkos::View <double*> vy_arr,
                    Kokkos::View <double*> vz_arr, double dt)
{
    short buffer[27];
    Kokkos::parallel_for(nparticles, 
    KOKKOS_LAMBDA (const int64_t index) 
    {
      x_arr(index) += vx_arr(index) * dt;
      y_arr(index) += vy_arr(index) * dt;
      z_arr(index) += vz_arr(index) * dt;
      tag_arr(index) = ComputeTag(myMeshBlock, x_arr(index),y_arr(index),z_arr(index));
      buffer[tag_arr(index)] += 1;
    });
}

void SortTag(const int64_t nparticles, Kokkos::View <size_t*> &tag_arr)
{

  return;
}