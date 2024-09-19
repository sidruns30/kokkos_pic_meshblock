// HH: see comment in main.cpp
//
// #include "../include/util.hpp"

#include "util.hpp"

// HH: it's also a good practice to include headers for all libraries you're using,
// ... even if they're already included in the header file
#include <algorithm>
#include <iostream>
#include <vector>

void PrintVec(std::vector<float> vec) {
  // HH: when the type can be deduced, just use `auto`
  auto nelements = std::min(10, (int)vec.size());

  // HH: better to use `auto` here too, unless you have a good reason (e.g., index goes negative)
  for (auto i = 0; i < nelements - 1; i++) {
    std::cout << vec[i] << "\t";
  }

  std::cout << vec[nelements - 1] << "\n";
}
