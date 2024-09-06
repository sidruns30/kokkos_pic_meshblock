#include "../include/particle_tracker.hpp"


void Particle::InitializeParticle(std::vector<float> xi, std::vector<float> vi)
{
    for (int i=0; i<3; i++)
    {
        this->pos[i] = xi[i];
        this->vel[i] = vi[i];
    }
    return;
}

// Just something to move particles around
void Particle::PushParticle(std::vector<float> acc, double dt)
{
    for (int i=0; i<3; i++)
    {
        this->pos[i] += this->vel[i] * dt;
        this->vel[i] += acc[i] * dt;
    }
    return;
}