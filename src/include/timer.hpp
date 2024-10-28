#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <iostream>
#include <Kokkos_Core.hpp>

#define TIMER_START(label) \
    Kokkos::fence(); \
    auto start_##label = std::chrono::high_resolution_clock::now();

#define TIMER_STOP(label) \
    Kokkos::fence(); \
    auto stop_##label = std::chrono::high_resolution_clock::now(); \
    auto duration_##label = std::chrono::duration_cast<std::chrono::microseconds>(stop_##label - start_##label).count(); \
    std::cout << "Timer [" #label "]: " << duration_##label << " microseconds" << std::endl;

#endif // TIMER_HPP