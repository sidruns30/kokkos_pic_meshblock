#include "particle_tracker.hpp"

#include "global.hpp"
#include "particle_tags.hpp"

#include <Kokkos_Core.hpp>

#include <array>
#include <iostream>
#include <random>

// Initialize the particle arrays
// Initialization is done on the host (CPU)
void InitializeParticleArrays(std::size_t               nparticles,
                              const std::array<int, 6>& meshblock,
                              Kokkos::View<short*>      tag_arr,
                              Kokkos::View<int*>        i_arr,
                              Kokkos::View<int*>        j_arr,
                              Kokkos::View<int*>        k_arr,
                              Kokkos::View<float*>      dx_arr,
                              Kokkos::View<float*>      dy_arr,
                              Kokkos::View<float*>      dz_arr,
                              Kokkos::View<real_t*>     vx_arr,
                              Kokkos::View<real_t*>     vy_arr,
                              Kokkos::View<real_t*>     vz_arr) {
  // Create Random Distributions to Initialize Particle Arrays
  std::uniform_real_distribution<> dis_x(meshblock[0], meshblock[1]);
  std::uniform_real_distribution<> dis_y(meshblock[2], meshblock[3]);
  std::uniform_real_distribution<> dis_z(meshblock[4], meshblock[5]);
  std::uniform_real_distribution<> dis_vel(-1, 1);
  std::random_device               rd;
  std::mt19937                     gen(rd());
  // mirror view

  auto i_arr_h  = Kokkos::create_mirror_view(i_arr);
  auto j_arr_h  = Kokkos::create_mirror_view(j_arr);
  auto k_arr_h  = Kokkos::create_mirror_view(k_arr);
  auto dx_arr_h = Kokkos::create_mirror_view(dx_arr);
  auto dy_arr_h = Kokkos::create_mirror_view(dy_arr);
  auto dz_arr_h = Kokkos::create_mirror_view(dz_arr);
  auto vx_arr_h = Kokkos::create_mirror_view(vx_arr);
  auto vy_arr_h = Kokkos::create_mirror_view(vy_arr);
  auto vz_arr_h = Kokkos::create_mirror_view(vz_arr);

  auto tag_arr_h = Kokkos::create_mirror_view(tag_arr);

  for (auto p = 0; p < nparticles; p++) {
    const auto x = dis_x(gen);
    i_arr_h(p)   = static_cast<int>(x);
    dx_arr_h(p)  = static_cast<float>(x) - static_cast<float>(i_arr_h(p));
    const auto y = dis_y(gen);
    j_arr_h(p)   = static_cast<int>(y);
    dy_arr_h(p)  = static_cast<float>(y) - static_cast<float>(j_arr_h(p));
    const auto z = dis_z(gen);
    k_arr_h(p)   = static_cast<int>(z);
    dz_arr_h(p)  = static_cast<float>(z) - static_cast<float>(k_arr_h(p));
    vx_arr_h(p)  = dis_vel(gen);
    vy_arr_h(p)  = dis_vel(gen);
    vz_arr_h(p)  = dis_vel(gen);

    tag_arr_h(p) = 1;
  }

  Kokkos::deep_copy(i_arr, i_arr_h);
  Kokkos::deep_copy(j_arr, j_arr_h);
  Kokkos::deep_copy(k_arr, k_arr_h);
  Kokkos::deep_copy(dx_arr, dx_arr_h);
  Kokkos::deep_copy(dy_arr, dy_arr_h);
  Kokkos::deep_copy(dz_arr, dz_arr_h);
  Kokkos::deep_copy(vx_arr, vx_arr_h);
  Kokkos::deep_copy(vy_arr, vy_arr_h);
  Kokkos::deep_copy(vz_arr, vz_arr_h);
  Kokkos::deep_copy(tag_arr, tag_arr_h);
  Kokkos::fence();
  std::cout << "Particles initialized\n";
}

// Particles are pushed on the device (GPU)
void PushParticles(std::size_t               nparticles,
                   const std::array<int, 6>& meshblock,
                   Kokkos::View<size_t[28]>  tag_ctr_arr,
                   Kokkos::View<short*>      tag_arr,
                   Kokkos::View<int*>        i_arr,
                   Kokkos::View<int*>        j_arr,
                   Kokkos::View<int*>        k_arr,
                   Kokkos::View<float*>      dx_arr,
                   Kokkos::View<float*>      dy_arr,
                   Kokkos::View<float*>      dz_arr,
                   Kokkos::View<real_t*>     vx_arr,
                   Kokkos::View<real_t*>     vy_arr,
                   Kokkos::View<real_t*>     vz_arr,
                   real_t                    dt) {
  const int xmin = meshblock[0];
  const int xmax = meshblock[1];
  const int ymin = meshblock[2];
  const int ymax = meshblock[3];
  const int zmin = meshblock[4];
  const int zmax = meshblock[5];

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
      const auto x_new = static_cast<real_t>(i_arr(p)) +
                         static_cast<real_t>(dx_arr(p)) + vx_arr(p) * dt;
      i_arr(p)  = static_cast<int>(x_new);
      dx_arr(p) = static_cast<float>(x_new) - static_cast<float>(i_arr(p));

      const auto y_new = static_cast<real_t>(j_arr(p)) +
                         static_cast<real_t>(dy_arr(p)) + vy_arr(p) * dt;
      j_arr(p)  = static_cast<int>(y_new);
      dy_arr(p) = static_cast<float>(y_new) - static_cast<float>(j_arr(p));

      const auto z_new = static_cast<real_t>(k_arr(p)) +
                         static_cast<real_t>(dz_arr(p)) + vz_arr(p) * dt;
      k_arr(p)  = static_cast<int>(z_new);
      dz_arr(p) = static_cast<float>(z_new) - static_cast<float>(k_arr(p));

      tag_arr(p) = SendTag(tag_arr(p),
                           i_arr(p) < xmin,
                           i_arr(p) >= xmax,
                           j_arr(p) < ymin,
                           j_arr(p) >= ymax,
                           k_arr(p) < zmin,
                           k_arr(p) >= zmax);
      Kokkos::atomic_increment(&tag_ctr_arr(tag_arr(p)));
    });
  Kokkos::fence();
  std::cout << "Particles pushed\n";
}

void PrintTags(std::size_t                   npart,
               Kokkos::View<std::size_t[28]> npart_per_tag,
               Kokkos::View<short*>          tags,
               bool                          print_ctrs) {
  auto tags_h = Kokkos::create_mirror_view(tags);

  Kokkos::deep_copy(tags_h, tags);

  for (auto i = 0; i < npart; i++) {
    std::cout << tags_h(i) << " ";
  }
  std::cout << std::endl << std::endl;
  if (print_ctrs) {
    auto npart_per_tag_h = Kokkos::create_mirror_view(npart_per_tag);
    Kokkos::deep_copy(npart_per_tag_h, npart_per_tag);
    std::cout << "Number of particles per tag:\n";
    for (auto i = 0; i < 28; i++) {
      std::cout << "Tag " << i << ": " << npart_per_tag_h(i) << std::endl;
    }
  }
}
