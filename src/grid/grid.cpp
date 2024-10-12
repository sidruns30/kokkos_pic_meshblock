#include "grid.hpp"

#include <iostream>

/*
    Return the Meshblock ID of a particle given its position
    Since most particles are going to be in the same MeshBlock, we
    want them to be evaluated the fastest.
    The particles on the faces should be evaluated the second fastest,
    edges the third fastest and corner casees at the very end
    Block:                                          Tag:
    x,y,z contained                                 0
    yz contained, x < xmin                          1
    yz contained, x > xmax                          2
    xz contained, y < ymin                          3
    xz contained, y > ymax                          4
    xy contained, z < zmin                          5
    xy contained, z > zmax                          6
    z contained, x < xmin, y < ymin                 7
    z contained, x < xmin, y > ymax                 8
    z contained, x > xmax, y < ymin                 9
    z contained, x > xmax, y > ymax                 10
    y contained, x < xmin, z < zmin                 11
    y contained, x < xmin, z > zmax                 12
    y contained, x > xmax, z < zmin                 13
    y contained, x > xmax, z > zmax                 14
    x contained, y < ymin, z < zmin                 15
    x contained, y < ymin, z > zmax                 16
    x contained, y > ymax, z < zmin                 17
    x contained, y > ymax, z > zmax                 18
    x < xmin, y < ymin, z < zmin                    19
    x < xmin, y < ymin, z > max                     20
    x < xmin, y > ymax, z < zmin                    21
    x < xmin, y > ymax, z > zmax                     22
    x > xmax, y < ymin, z < zmin                    23
    x > xmax, y < ymin, z > zmax                     24
    x > xmax, y > ymax, z < zmin                    25
    x > xmax, y > ymax, z > max                     26
*/