#ifndef GRID_H
#define GRID_H

#include "util.hpp"

#include <iostream>

/*
  SS: Commenting this out for now since I don't know how to send
      the enum to the GPU
enum IndexPosition {
  XYZ          = 0,
  YZXmin       = 1,
  YZXmax       = 2,
  XZYmin       = 3,
  XZYmax       = 4,
  XYZmin       = 5,
  XYZmax       = 6,
  ZXminYmin    = 7,
  ZXminYmax    = 8,
  ZXmaxYmin    = 9,
  ZXmaxYmax    = 10,
  YXminZmin    = 11,
  YXminZmax    = 12,
  YXmaxZmin    = 13,
  YXmaxZmax    = 14,
  XYminZmin    = 15,
  XYminZmax    = 16,
  XYmaxZmin    = 17,
  XYmaxZmax    = 18,
  XminYminZmin = 19,
  XminYminZmax = 20,
  XminYmaxZmin = 21,
  XminYmaxZmax = 22,
  XmaxYminZmin = 23,
  XmaxYminZmax = 24,
  XmaxYmaxZmin = 25,
  XmaxYmaxZmax = 26,
  BDRY_OUT     = 27
};
*/


struct MeshBlock {
  float xmin, xmax, ymin, ymax, zmin, zmax;
  float dx, dy, dz;

  MeshBlock(std::vector<float> xi_min_mb,
            std::vector<float> dxi_mb,
            std::vector<int>   nxi_cells)
    : xmin(xi_min_mb[0])
    , xmax(xi_min_mb[0] + dxi_mb[0])
    , ymin(xi_min_mb[1])
    , ymax(xi_min_mb[1] + dxi_mb[1])
    , zmin(xi_min_mb[2])
    , zmax(xi_min_mb[2] + dxi_mb[2])
    , dx(dxi_mb[0] / nxi_cells[0])
    , dy(dxi_mb[1] / nxi_cells[1])
    , dz(dxi_mb[2] / nxi_cells[2]) {
    std::cout << "MeshBlock Initialized" << std::endl;
  }

};

#endif
