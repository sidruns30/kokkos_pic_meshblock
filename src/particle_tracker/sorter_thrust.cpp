#include "sorter_thrust.hpp"

void SortThrust(std::size_t           nparticles,
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
  TIMER_START(SortThrust);

  // Create a tuple of views
  auto tuple = thrust::make_zip_iterator(thrust::make_tuple(KE::begin(tag_arr),
                                                            KE::begin(i_arr),
                                                            KE::begin(j_arr),
                                                            KE::begin(k_arr),
                                                            KE::begin(dx_arr),
                                                            KE::begin(dy_arr),
                                                            KE::begin(dz_arr),
                                                            KE::begin(vx_arr),
                                                            KE::begin(vy_arr),
                                                            KE::begin(vz_arr)));

  thrust::sort_by_key(thrust::device, KE::begin(tag_arr), KE::end(tag_arr), tuple);

  TIMER_STOP(SortThrust);
  std::cout << "Particles sorted\n";
  return;
}
