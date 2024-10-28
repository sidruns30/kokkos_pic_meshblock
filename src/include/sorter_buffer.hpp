#ifndef SORTER_BUFFER_H
#define SORTER_BUFFER_H

#include "global.hpp"
#include "timer.hpp"
#include <Kokkos_Core.hpp>

void SortBuffer(std::size_t                   ,
                Kokkos::View<short*>          ,
                const Kokkos::View<size_t[28]>,
                Kokkos::View<int*>            ,
                Kokkos::View<int*>            ,
                Kokkos::View<int*>            ,
                Kokkos::View<float*>          ,
                Kokkos::View<float*>          ,
                Kokkos::View<float*>          ,
                Kokkos::View<real_t*>         ,
                Kokkos::View<real_t*>         ,
                Kokkos::View<real_t*>         );

#endif // SORTER_BUFFER_H