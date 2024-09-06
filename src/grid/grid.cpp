#include "../include/grid.hpp"

// We want to divide the grid into boxes of equal volumes
// For now, let us only divide the grid based on the x coords
void CreateMeshblockBoundaries(float xmin, float xmax, float ymin, float ymax, 
                               float zmin, float zmax, int nblocks, std::vector <float> &xmb,
                               std::vector <float> &ymb, std::vector <float> &zmb)
{
    int nblocks_x, nblocks_y, nblocks_z;
    nblocks_x = nblocks;
    nblocks_y = 1;
    nblocks_z = 1;
    float delta_x_mb = (xmax - xmin) / nblocks_x;
    float delta_y_mb = (ymax - ymin) / nblocks_y;
    float delta_z_mb = (zmax - zmin) / nblocks_z;

    for (int i=0; i<nblocks_x+1; i++) xmb.push_back(i * delta_x_mb);
    for (int i=0; i<nblocks_y+1; i++) ymb.push_back(i * delta_y_mb);
    for (int i=0; i<nblocks_z+1; i++) zmb.push_back(i * delta_z_mb);
    return;
}


int ComputeRank(const std::vector<float> &pos, const std::vector <float> &xmb,
                const std::vector <float> &ymb, const std::vector <float> &zmb)
{
    int i_xmb, i_ymb, i_zmb;
    int nblocks_x, nblocks_y, nblocks_z;
    nblocks_x = xmb.size();
    nblocks_y = ymb.size();
    nblocks_z = zmb.size();

    // Find out which block the cell
    for (int i_xmb=0; i_xmb<nblocks_x; i_xmb++)
    {
        if (xmb[i_xmb] > pos[0]) break; 
    }
    for (int i_ymb=0; i_ymb<nblocks_y; i_ymb++)
    {
        if (ymb[i_ymb] > pos[1]) break; 
    }
        for (int i_zmb=0; i_zmb<nblocks_z; i_zmb++)
    {
        if (zmb[i_zmb] > pos[2]) break; 
    }

    return i_zmb * nblocks_y * nblocks_x + i_ymb * nblocks_x + i_xmb;
}