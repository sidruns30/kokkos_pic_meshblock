#include "sorter.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <chrono>
#include <utility>

void Sort(std::size_t           nparticles,
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
  auto start              = std::chrono::high_resolution_clock::now();
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

  Kokkos::fence();
  auto end1 = std::chrono::high_resolution_clock::now();

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
  // sort others ..
  Kokkos::fence();
  auto end2 = std::chrono::high_resolution_clock::now();
  std::cout
    << "Permute vector time: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start).count()
    << " ms\n";
  std::cout
    << "Sorting time: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count()
    << " ms\n";
  std::cout << "Particles sorted\n";
}
