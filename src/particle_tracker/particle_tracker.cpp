// HH: see comment in main.cpp

// #include "../include/particle_tracker.hpp"

#include "particle_tracker.hpp"

// HH: see comment in util.cpp

#include <Kokkos_Core.hpp>

#include <random>

// HH: see my comments in particle_tracker.hpp
// ... since you're not modifying the meshblock, pass it by const reference
//
// void InitializeParticleArrays(const int64_t          nparticles,
//                               MeshBlock&             myMeshBlock,
//                               Kokkos::View<size_t*>& tag_arr,
//                               Kokkos::View<double*>& x_arr,
//                               Kokkos::View<double*>& y_arr,
//                               Kokkos::View<double*>& z_arr,
//                               Kokkos::View<double*>& vx_arr,
//                               Kokkos::View<double*>& vy_arr,
//                               Kokkos::View<double*>& vz_arr) {
//
// Initialize the particle arrays
void InitializeParticleArrays(std::size_t           nparticles,
                              const MeshBlock&      myMeshBlock,
                              Kokkos::View<short*>  tag_arr,
                              Kokkos::View<double*> x_arr,
                              Kokkos::View<double*> y_arr,
                              Kokkos::View<double*> z_arr,
                              Kokkos::View<double*> vx_arr,
                              Kokkos::View<double*> vy_arr,
                              Kokkos::View<double*> vz_arr) {
  // Create Random Distributions to Initialize Particle Arrays
  std::uniform_real_distribution<> dis_x(myMeshBlock.xmin, myMeshBlock.xmax);
  std::uniform_real_distribution<> dis_y(myMeshBlock.ymin, myMeshBlock.ymax);
  std::uniform_real_distribution<> dis_z(myMeshBlock.zmin, myMeshBlock.zmax);
  std::uniform_real_distribution<> dis_vel(-1, 1);
  std::random_device               rd;
  std::mt19937                     gen(rd());
  // mirror view
  // Kokkos for rng

  auto x_arr_h   = Kokkos::create_mirror_view(x_arr);
  auto y_arr_h   = Kokkos::create_mirror_view(y_arr);
  auto z_arr_h   = Kokkos::create_mirror_view(z_arr);
  auto vx_arr_h  = Kokkos::create_mirror_view(vx_arr);
  auto vy_arr_h  = Kokkos::create_mirror_view(vy_arr);
  auto vz_arr_h  = Kokkos::create_mirror_view(vz_arr);
  auto tag_arr_h = Kokkos::create_mirror_view(tag_arr);

  // HH: see my comment in util.cpp
  // ... I also usually use a shorter name for the index, since it's used a lot
  //
  // for (int64_t index = 0; index < nparticles; index++) {
  for (auto p = 0; p < nparticles; p++) {
    x_arr_h(p)   = dis_x(gen);
    y_arr_h(p)   = dis_y(gen);
    z_arr_h(p)   = dis_z(gen);
    vx_arr_h(p)  = dis_vel(gen);
    vy_arr_h(p)  = dis_vel(gen);
    vz_arr_h(p)  = dis_vel(gen);
    tag_arr_h(p) = ComputeTag(myMeshBlock, x_arr_h(p), y_arr_h(p), z_arr_h(p));
  }

  Kokkos::deep_copy(x_arr, x_arr_h);
  Kokkos::deep_copy(y_arr, y_arr_h);
  Kokkos::deep_copy(z_arr, z_arr_h);
  Kokkos::deep_copy(vx_arr, vx_arr_h);
  Kokkos::deep_copy(vy_arr, vy_arr_h);
  Kokkos::deep_copy(vz_arr, vz_arr_h);
  Kokkos::deep_copy(tag_arr, tag_arr_h);
}

// HH: see my comments in particle_tracker.hpp
//
// void PushParticles(const int64_t         nparticles,
//                    const MeshBlock       myMeshBlock,
//                    Kokkos::View<size_t*> tag_arr,
//                    Kokkos::View<double*> x_arr,
//                    Kokkos::View<double*> y_arr,
//                    Kokkos::View<double*> z_arr,
//                    Kokkos::View<double*> vx_arr,
//                    Kokkos::View<double*> vy_arr,
//                    Kokkos::View<double*> vz_arr,
//                    double                dt) {
void PushParticles(std::size_t           nparticles,
                   const MeshBlock&      myMeshBlock,
                   Kokkos::View<short*>  tag_arr,
                   Kokkos::View<double*> x_arr,
                   Kokkos::View<double*> y_arr,
                   Kokkos::View<double*> z_arr,
                   Kokkos::View<double*> vx_arr,
                   Kokkos::View<double*> vy_arr,
                   Kokkos::View<double*> vz_arr,
                   double                dt) {
  // HH: see my comment above about the index
  //
  // Kokkos::parallel_for(
  //   nparticles,
  //   KOKKOS_LAMBDA(const int64_t index) {
  //     x_arr(index) += vx_arr(index) * dt;
  //     y_arr(index) += vy_arr(index) * dt;
  //     z_arr(index) += vz_arr(index) * dt;
  //     tag_arr(
  //       index) = ComputeTag(myMeshBlock, x_arr(index), y_arr(index), z_arr(index));
  //   });
  Kokkos::parallel_for(
    nparticles,
    KOKKOS_LAMBDA(const std::size_t p) {
      x_arr(p) += vx_arr(p) * dt;
      y_arr(p) += vy_arr(p) * dt;
      z_arr(p) += vz_arr(p) * dt;

      // HH:  this will not work on GPU, unless ComputeTag is a KOKKOS_INLINE_FUNCTION
      // and myMeshBlock is defined on the GPU (which right now it is not)
      tag_arr(p) = ComputeTag(myMeshBlock, x_arr(p), y_arr(p), z_arr(p));
    });
}

void SortTag(std::size_t nparticles, Kokkos::View<short*> tag_arr) {

  return;
}
