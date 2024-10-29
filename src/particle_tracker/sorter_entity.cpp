#include "sorter_entity.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <chrono>
#include <utility>

void SortEntity(std::size_t           nparticles,
          Kokkos::View<short*>  tag_arr,
          Kokkos::View<int*>    i_arr,
          Kokkos::View<int*>    j_arr,
          Kokkos::View<int*>    k_arr,
          Kokkos::View<float*>  dx_arr,
          Kokkos::View<float*>  dy_arr,
          Kokkos::View<float*>  dz_arr,
          Kokkos::View<real_t*> vx_arr,
          Kokkos::View<real_t*> vy_arr,
          Kokkos::View<real_t*> vz_arr) {
  TIMER_START(SortEntity);
  using KeyType           = Kokkos::View<short*>;
  using BinOp             = BinTag<KeyType>;
  const std::size_t ntags = 28;
  BinOp             bin_op(ntags);
  using range_tuple_t                   = std::pair<std::size_t, std::size_t>;
  auto                            slice = range_tuple_t(0, nparticles);
  Kokkos::BinSort<KeyType, BinOp> Sorter(Kokkos::subview(tag_arr, slice),
                                         bin_op,
                                         false);
  Sorter.create_permute_vector();

  Sorter.sort(Kokkos::subview(tag_arr, slice));
  Sorter.sort(Kokkos::subview(i_arr, slice));
  Sorter.sort(Kokkos::subview(j_arr, slice));
  Sorter.sort(Kokkos::subview(k_arr, slice));
  Sorter.sort(Kokkos::subview(dx_arr, slice));
  Sorter.sort(Kokkos::subview(dy_arr, slice));
  Sorter.sort(Kokkos::subview(dz_arr, slice));
  Sorter.sort(Kokkos::subview(vx_arr, slice));
  Sorter.sort(Kokkos::subview(vy_arr, slice));
  Sorter.sort(Kokkos::subview(vz_arr, slice));
  TIMER_STOP(SortEntity);
  std::cout << "Particles sorted\n";
  return;
}
