#ifndef SORTER_THRUST_H
#define SORTER_THRUST_H

#include "global.hpp"
#include "timer.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <tuple>

void SortThrust(std::size_t                   ,
                Kokkos::View<short*>          ,
                Kokkos::View<int*>            ,
                Kokkos::View<int*>            ,
                Kokkos::View<int*>            ,
                Kokkos::View<float*>          ,
                Kokkos::View<float*>          ,
                Kokkos::View<float*>          ,
                Kokkos::View<real_t*>         ,
                Kokkos::View<real_t*>         ,
                Kokkos::View<real_t*>         );


#endif // SORTER_THRUST_H