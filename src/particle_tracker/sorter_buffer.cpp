#include "sorter_buffer.hpp"

#include <Kokkos_Core.hpp>

/*
    SS: Sort the particles based on their tags,
        making use of the 'tag_ctr_arr', which
        contains the number of particles per tag.
        We first allocate buffers based on the
        count per tag and then populate the buffers
        by traversing through the array
*/
void SortBuffer(std::size_t                     nparticles,
                Kokkos::View<short*>            tag_arr,
                const Kokkos::View<size_t[28]>  tag_ctr_arr,
                Kokkos::View<int*>              i_arr,
                Kokkos::View<int*>              j_arr,
                Kokkos::View<int*>              k_arr,
                Kokkos::View<float*>            dx_arr,
                Kokkos::View<float*>            dy_arr,
                Kokkos::View<float*>            dz_arr,
                Kokkos::View<real_t*>           vx_arr,
                Kokkos::View<real_t*>           vy_arr,
                Kokkos::View<real_t*>           vz_arr)
{         
  TIMER_START(SortBuffer);

  // Allocate buffers
  Kokkos::View<short*>      tag_buffer("tag_buffer", nparticles);
  Kokkos::View<int*>        i_buffer("i_buffer", nparticles);
  Kokkos::View<int*>        j_buffer("j_buffer", nparticles);
  Kokkos::View<int*>        k_buffer("k_buffer", nparticles);
  Kokkos::View<float*>      dx_buffer("dx_buffer", nparticles);
  Kokkos::View<float*>      dy_buffer("dy_buffer", nparticles);
  Kokkos::View<float*>      dz_buffer("dz_buffer", nparticles);
  Kokkos::View<real_t*>     vx_buffer("vx_buffer", nparticles);
  Kokkos::View<real_t*>     vy_buffer("vy_buffer", nparticles);
  Kokkos::View<real_t*>     vz_buffer("vz_buffer", nparticles);

  // Make a cumulative array of the tag counts
  Kokkos::View<size_t[29]> tag_cumulative("tag_cumulative", 28);
  Kokkos::parallel_scan(
    "Tag cumulative scan",
    Kokkos::RangePolicy<Kokkos::Cuda>(0, 28),
    KOKKOS_LAMBDA(const std::size_t i, size_t& update, const bool final) {
      update += tag_ctr_arr(i);
      if (final) {
        tag_cumulative(i + 1) = update;
      }
    });

  // Array to keep track of number of elements initialized in the buffer
  Kokkos::View<size_t[28]>       buffer_ctr("buffer_ctr", nparticles);

  // Begin populating
  Kokkos::parallel_for(
    "Particle pusher loop",
    Kokkos::RangePolicy<Kokkos::Cuda>(0, nparticles),
    KOKKOS_LAMBDA(const std::size_t p) {

    const auto tag            = tag_arr(p);
    const auto local_index    = Kokkos::atomic_fetch_add(&buffer_ctr(tag), 1);
    const auto global_index   = tag_cumulative(tag) + local_index;

    tag_buffer(global_index)  = tag_arr(p);
    i_buffer(global_index)    = i_arr(p);
    j_buffer(global_index)    = j_arr(p);
    k_buffer(global_index)    = k_arr(p);
    dx_buffer(global_index)   = dx_arr(p);
    dy_buffer(global_index)   = dy_arr(p);
    dz_buffer(global_index)   = dz_arr(p);
    vx_buffer(global_index)   = vx_arr(p);
    vy_buffer(global_index)   = vy_arr(p);
    vz_buffer(global_index)   = vz_arr(p);
    });

  // SS: A little danger here since the memory pointing to the original
  //     arrays is not freed.
  // Copy back to original arrays
  tag_arr                     = tag_buffer;
  i_arr                       = i_buffer;
  j_arr                       = j_buffer;
  k_arr                       = k_buffer;
  dx_arr                      = dx_buffer;
  dy_arr                      = dy_buffer;
  dz_arr                      = dz_buffer;
  vx_arr                      = vx_buffer;
  vy_arr                      = vy_buffer;
  vz_arr                      = vz_buffer;

  TIMER_STOP(SortBuffer);
  std::cout << "Particles sorted\n";
  return;
}