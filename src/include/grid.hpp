#include "util.hpp"
#include "particle_tracker.hpp"

#ifndef GRID_H
    #define GRID_H
    void CreateMeshblockBoundaries(float xmin, float xmax, float ymin, float ymax, 
                                float zmin, float zmax, int nblocks, std::vector <float> &xmb,
                                std::vector <float> &ymb, std::vector <float> &zmb);
    int ComputeRank(const std::vector<float> &pos, const std::vector <float> &xmb,
                    const std::vector <float> &ymb, const std::vector <float> &zmb);
    class Meshblock
    {
        public:
            int rank;
            std::vector <float> xmb, ymb, zmb;
            std::vector <Particle> local_particles;
            Meshblock() {};
            void PushParticles();
    };
#endif