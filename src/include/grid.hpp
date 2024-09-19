#include "util.hpp"

#ifndef GRID_H
    #define GRID_H

enum IndexPosition
{
    XYZ = 0,
    YZXmin = 1,
    YZXmax = 2,
    XZYmin = 3,
    XZYmax = 4,
    XYZmin = 5,
    XYZmax = 6,
    ZXminYmin = 7,
    ZXminYmax = 8,
    ZXmaxYmin = 9,
    ZXmaxYmax = 10,
    YXminZmin = 11,
    YXminZmax = 12,
    YXmaxZmin = 13,
    YXmaxZmax = 14,
    XYminZmin = 15,
    XYminZmax = 16,
    XYmaxZmin = 17,
    XYmaxZmax = 18,
    XminYminZmin = 19,
    XminYminZmax = 20,
    XminYmaxZmin = 21,
    XminYmaxZmax = 22,
    XmaxYminZmin = 23,
    XmaxYminZmax = 24,
    XmaxYmaxZmin = 25,
    XmaxYmaxZmax = 26,
    BDRY_OUT = 27
};

/*
    (1) Create a MeshBlock and cells that are defined by
        xi_min_mb, dxi_mb; i=(1,2,3)
    (2) Create neighboring Meshblocks defined by:
        faces, edges, vertices
        xy_z_lower, xy_z_upper, yz_x_lower, 
        yz_x_upper, xz_y_lower, xz_y_upper
        x_y_lower_z_lower, x_y_lower_z_upper, 
        x_y_upper_z_lower, x_y_upper_z_upper, ...
        (6 + 8 + 8 = 22 neighbors)
    (3) Add functions to traverse through the cell given
        a Meshblock
*/

    class MeshBlock
{
    public:
        float xmin, xmax, ymin, ymax, zmin, zmax;
        float dx, dy, dz;
        MeshBlock(  std::vector <float> xi_min_mb, std::vector <float> dxi_mb,
                    std::vector <int> nxi_cells): 
                    xmin(xi_min_mb[0]), xmax(xi_min_mb[0] + dxi_mb[0]),
                    ymin(xi_min_mb[1]), ymax(xi_min_mb[1] + dxi_mb[1]),
                    zmin(xi_min_mb[2]), zmax(xi_min_mb[2] + dxi_mb[2]),
                    dx(dxi_mb[0]/nxi_cells[0]), dy(dxi_mb[1]/nxi_cells[1]), 
                    dz(dxi_mb[2]/nxi_cells[2])
                    {std::cout << "MeshBlock Initialized" << std::endl;}
        int ComputeTag(float x, float y, float z) const;
}
;

const int ComputeTag(const MeshBlock myMeshBlock, const float x, const float y, const float z);
#endif