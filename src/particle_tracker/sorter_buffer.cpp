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
void SortBuffer(std::size_t                    nparticles,
                Kokkos::View<short*>           tag_arr,
                const Kokkos::View<size_t[29]> tag_ctr_cumsum,
                Kokkos::View<int*>             i_arr,
                Kokkos::View<int*>             j_arr,
                Kokkos::View<int*>             k_arr,
                Kokkos::View<float*>           dx_arr,
                Kokkos::View<float*>           dy_arr,
                Kokkos::View<float*>           dz_arr,
                Kokkos::View<real_t*>          vx_arr,
                Kokkos::View<real_t*>          vy_arr,
                Kokkos::View<real_t*>          vz_arr) {
  auto tag_ctr_cumsum_h = Kokkos::create_mirror_view(tag_ctr_cumsum);
  Kokkos::deep_copy(tag_ctr_cumsum_h, tag_ctr_cumsum);
  const auto buffsize = tag_ctr_cumsum_h(28) - tag_ctr_cumsum_h(2);
  // const auto buffsize = nparticles;
  std::cout << "Buffer size: " << buffsize << std::endl;

  TIMER_START(SortBuffer);

  // Allocate buffers
  Kokkos::View<short*>  tag_buffer("tag_buffer", buffsize);
  Kokkos::View<int*>    i_buffer("i_buffer", buffsize);
  Kokkos::View<int*>    j_buffer("j_buffer", buffsize);
  Kokkos::View<int*>    k_buffer("k_buffer", buffsize);
  Kokkos::View<float*>  dx_buffer("dx_buffer", buffsize);
  Kokkos::View<float*>  dy_buffer("dy_buffer", buffsize);
  Kokkos::View<float*>  dz_buffer("dz_buffer", buffsize);
  Kokkos::View<real_t*> vx_buffer("vx_buffer", buffsize);
  Kokkos::View<real_t*> vy_buffer("vy_buffer", buffsize);
  Kokkos::View<real_t*> vz_buffer("vz_buffer", buffsize);

  // Array to keep track of number of elements initialized in the buffer
  Kokkos::View<size_t[28]> buffer_ctr("buffer_ctr");

  // Begin populating
  Kokkos::parallel_for(
    "Particle pusher loop",
    Kokkos::RangePolicy<Kokkos::Cuda>(0, nparticles),
    KOKKOS_LAMBDA(const std::size_t p) {
      if (tag_arr(p) < 2) {
        return;
      }
      const auto tag          = tag_arr(p);
      const auto local_index  = Kokkos::atomic_fetch_add(&buffer_ctr(tag), 1);
      // const auto global_index = tag_ctr_cumsum(tag) + local_index;
      const auto global_index = tag_ctr_cumsum(tag) - tag_ctr_cumsum(2) +
                                local_index;

      tag_buffer(global_index) = tag;
      i_buffer(global_index)   = i_arr(p);
      j_buffer(global_index)   = j_arr(p);
      k_buffer(global_index)   = k_arr(p);
      dx_buffer(global_index)  = dx_arr(p);
      dy_buffer(global_index)  = dy_arr(p);
      dz_buffer(global_index)  = dz_arr(p);
      vx_buffer(global_index)  = vx_arr(p);
      vy_buffer(global_index)  = vy_arr(p);
      vz_buffer(global_index)  = vz_arr(p);
    });

  // Copy elements from buffer arrays to main arrays
  /*
  Kokkos::parallel_for(
    "Copy buffer to main arrays",
    Kokkos::RangePolicy<Kokkos::Cuda>(0, nparticles),
    KOKKOS_LAMBDA(const std::size_t p) {
      tag_arr(p) = tag_buffer(p);
      i_arr(p)   = i_buffer(p);
      j_arr(p)   = j_buffer(p);
      k_arr(p)   = k_buffer(p);
      dx_arr(p)  = dx_buffer(p);
      dy_arr(p)  = dy_buffer(p);
      dz_arr(p)  = dz_buffer(p);
      vx_arr(p)  = vx_buffer(p);
      vy_arr(p)  = vy_buffer(p);
      vz_arr(p)  = vz_buffer(p);
    });
  */

  // Copy back to original arrays
  // Kokkos::deep_copy(tag_arr, tag_buffer);
  // Kokkos::deep_copy(i_arr, i_buffer);
  // Kokkos::deep_copy(j_arr, j_buffer);
  // Kokkos::deep_copy(k_arr, k_buffer);
  // Kokkos::deep_copy(dx_arr, dx_buffer);
  // Kokkos::deep_copy(dy_arr, dy_buffer);
  // Kokkos::deep_copy(dz_arr, dz_buffer);
  // Kokkos::deep_copy(vx_arr, vx_buffer);
  // Kokkos::deep_copy(vy_arr, vy_buffer);
  // Kokkos::deep_copy(vz_arr, vz_buffer);

  TIMER_STOP(SortBuffer);
  std::cout << "Particles sorted\n";
  return;
}
