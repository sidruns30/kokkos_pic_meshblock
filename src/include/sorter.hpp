#ifndef SORTER_H
#define SORTER_H

#include "global.hpp"

#include <Kokkos_Core.hpp>

template <class KeyViewType>
struct BinTag {
  BinTag(const int& max_bins) : m_max_bins { max_bins } {}

  template <class ViewType>
  KOKKOS_INLINE_FUNCTION auto bin(ViewType& keys, const int& i) const -> int {
    return (keys(i) == 0) ? 1 : ((keys(i) == 1) ? 0 : keys(i));
  }

  KOKKOS_INLINE_FUNCTION auto max_bins() const -> int {
    return m_max_bins;
  }

  template <class ViewType, typename iT1, typename iT2>
  KOKKOS_INLINE_FUNCTION auto operator()(ViewType&, iT1&, iT2&) const -> bool {
    return false;
  }

private:
  const int m_max_bins;
};

void Sort(std::size_t,
          Kokkos::View<short*>,
          Kokkos::View<int*>,
          Kokkos::View<int*>,
          Kokkos::View<int*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<float*>,
          Kokkos::View<real_t*>,
          Kokkos::View<real_t*>,
          Kokkos::View<real_t*>);

#endif
