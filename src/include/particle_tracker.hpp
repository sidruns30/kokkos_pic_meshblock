#include <random>
#ifndef PARTICLE_TRACKER_H
    #define PARTICLE_TRACKER_H
    class Particle
    {
        public:
            std::vector <float> pos {0., 0., 0.};
            std::vector <float> vel {0., 0., 0.};
            int rank;
            void InitializeParticle(std::vector<float> xi, std::vector<float> vi);
            void PushParticle(std::vector<float> acc, double dt);
        Particle()
        {
            rank = 0;
        }

    };
#endif