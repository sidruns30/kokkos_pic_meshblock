#include "../include/grid.hpp"



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


int MeshBlock::ComputeTag(float x, float y, float z) const
{
    int tag;
    // z contained
    if (z > this-> zmin & z < this->zmax)
    {
        // y-z contained
        if (y > this->ymin & y < this->ymax)
        {
            // x, y, z contained
            if (x > this-> xmin & x < this->xmax)  {return XYZ;}
            // yz contained, x < xmin
            else if (x < this->xmin) {return YZXmin;}
            // yz contained, x > xmax
            else {return YZXmax;}
        }
        // x-z contained
        else if (x > this->xmin & x < this->xmax)
        {
            // xz contained, y < min
            if (y < this->ymin) {return XZYmin;}
            // xz contained, y > ymax
            else {return XZYmax;}
        }
    }
    // x contained, z not conatined
    else if (x > this->xmin & x < this->xmax)
    {
        // x-y contained
        if (y > this->ymin & y < this->ymax)
        {
            // x-y contained, z < zmin
            if (z < this->zmin) {return XYZmin;}
            else {return XYZmax;}
        }
    }

    // The particle now lies at least in the edge or the corner MeshBlock
    // Let us do the edges first
    // We know that the particle does not have x or z coordinate in the block

    // z contained only
    if (z > this->zmin & z < this->zmax)
    {
        if (x < this->xmin)
        {
            if (y < this->ymin) {return ZXminYmin;}
            else {return ZXminYmax;}
        }
        else
        {
            if (y < this->ymin) {return ZXmaxYmin;}
            else {return ZXmaxYmax;}
        }
    }
    // y contained only
    else if (y > this-> ymin & y < this->ymax)
    {
        if (x < this->xmin)
        {
            if (z < this->zmin) {return YXminZmin;}
            else {return YXminZmax;}
        }
        else
        {
            if (z < this->zmin) {return YXmaxZmin;}
            else {return YXmaxZmax;}
        }

    }
    // x contained only
    else if (x > this-> xmin & x < this->xmax)
    {
        if (y < this->ymin)
        {
            if (z < this->zmin) {return XYminZmin;}
            else {return XYminZmax;}
        }
        else
        {
            if (z < this->zmin) {return XYmaxZmin;}
            else {return XYmaxZmax;}
        }
    }
    //  In the cases below, the particle contains none of the coordinates
    //  so it must be in one of the corner cells
    else
    {
        if (x < this-> xmin)
        {
            if (y < this-> ymin)
            {
                if (z < this->zmin) {return XminYminZmin;}
                else {return XminYminZmax;}
            }
            else
            {
                if (z < this-> zmin) {return XminYmaxZmin;}
                else {return XminYmaxZmax;}
            }
        }
        else
        {
            if (y < this-> ymin)
            {
                if (z < this->zmin) {return XmaxYminZmin;}
                else {return XmaxYminZmax;}
            }
            else
            {
                if (z < this-> zmin) {return XmaxYmaxZmin;}
                else {return XmaxYmaxZmax;}
            }
        }
    }
}

const int ComputeTag(const MeshBlock myMeshBlock, const float x, const float y, const float z)
{
    int tag;
    // z contained
    if (z > myMeshBlock.zmin & z < myMeshBlock.zmax)
    {
        // y-z contained
        if (y > myMeshBlock.ymin & y < myMeshBlock.ymax)
        {
            // x, y, z contained
            if (x > myMeshBlock. xmin & x < myMeshBlock.xmax)  {return XYZ;}
            // yz contained, x < xmin
            else if (x < myMeshBlock.xmin) {return YZXmin;}
            // yz contained, x > xmax
            else {return YZXmax;}
        }
        // x-z contained
        else if (x > myMeshBlock.xmin & x < myMeshBlock.xmax)
        {
            // xz contained, y < min
            if (y < myMeshBlock.ymin) {return XZYmin;}
            // xz contained, y > ymax
            else {return XZYmax;}
        }
    }
    // x contained, z not conatined
    else if (x > myMeshBlock.xmin & x < myMeshBlock.xmax)
    {
        // x-y contained
        if (y > myMeshBlock.ymin & y < myMeshBlock.ymax)
        {
            // x-y contained, z < zmin
            if (z < myMeshBlock.zmin) {return XYZmin;}
            else {return XYZmax;}
        }
    }

    // The particle now lies at least in the edge or the corner MeshBlock
    // Let us do the edges first
    // We know that the particle does not have x or z coordinate in the block

    // z contained only
    if (z > myMeshBlock.zmin & z < myMeshBlock.zmax)
    {
        if (x < myMeshBlock.xmin)
        {
            if (y < myMeshBlock.ymin) {return ZXminYmin;}
            else {return ZXminYmax;}
        }
        else
        {
            if (y < myMeshBlock.ymin) {return ZXmaxYmin;}
            else {return ZXmaxYmax;}
        }
    }
    // y contained only
    else if (y > myMeshBlock. ymin & y < myMeshBlock.ymax)
    {
        if (x < myMeshBlock.xmin)
        {
            if (z < myMeshBlock.zmin) {return YXminZmin;}
            else {return YXminZmax;}
        }
        else
        {
            if (z < myMeshBlock.zmin) {return YXmaxZmin;}
            else {return YXmaxZmax;}
        }

    }
    // x contained only
    else if (x > myMeshBlock. xmin & x < myMeshBlock.xmax)
    {
        if (y < myMeshBlock.ymin)
        {
            if (z < myMeshBlock.zmin) {return XYminZmin;}
            else {return XYminZmax;}
        }
        else
        {
            if (z < myMeshBlock.zmin) {return XYmaxZmin;}
            else {return XYmaxZmax;}
        }
    }
    //  In the cases below, the particle contains none of the coordinates
    //  so it must be in one of the corner cells
    else
    {
        if (x < myMeshBlock. xmin)
        {
            if (y < myMeshBlock. ymin)
            {
                if (z < myMeshBlock.zmin) {return XminYminZmin;}
                else {return XminYminZmax;}
            }
            else
            {
                if (z < myMeshBlock. zmin) {return XminYmaxZmin;}
                else {return XminYmaxZmax;}
            }
        }
        else
        {
            if (y < myMeshBlock. ymin)
            {
                if (z < myMeshBlock.zmin) {return XmaxYminZmin;}
                else {return XmaxYminZmax;}
            }
            else
            {
                if (z < myMeshBlock. zmin) {return XmaxYmaxZmin;}
                else {return XmaxYmaxZmax;}
            }
        }
    }
}
