#include <Kokkos_Core.hpp>
#include "../include/util.hpp"
#include "../include/grid.hpp"
#include "../include/particle_tracker.hpp"


auto main(int argc, char* argv[]) -> int 
{
  // Set boundaries to the grid
  float xmin = 0.;
  float xmax = 1.;
  float ymin = 0.;
  float ymax = 1.;
  float zmin = 0.;
  float zmax = 1.;
  int nprocs = atoi(argv[1]);
  int rank;

  // Create arrays for meshblock boundaries
  std::vector <float> xmb, ymb, zmb;
  CreateMeshblockBoundaries(xmin, xmax, ymin, ymax, zmin, zmax, nprocs, xmb, ymb, zmb);

  // MPI region begins
  Kokkos::initialize(argc, argv);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "Number of processes available are " << nprocs << " and rank is " << rank << std::endl;

  // Initialize a few particles
  const int nparticles = 1;
  std::vector <Particle> all_particles;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_x(xmb[rank], xmb[rank+1]);
  std::uniform_real_distribution<> dis_y(ymb[rank], ymb[rank+1]);
  std::uniform_real_distribution<> dis_z(zmb[rank], zmb[rank+1]);
  std::uniform_real_distribution<> dis_vel(-1, 1);

  for (int i=0; i<nparticles; i++)
  {
    Particle p;
    float x = dis_x(gen);
    float y = dis_y(gen);
    float z = dis_z(gen);
    float vx = dis_vel(gen);
    float vy = dis_vel(gen);
    float vz = dis_vel(gen);
    std::vector <float> pos = {x, y, z};
    std::vector <float> vel = {vx, vy, vz};
    p.InitializeParticle(pos, vel);
    all_particles.push_back(p);
    PrintVec(pos);
  }


  //int rank;
  //std::cout << "Communicating \n";

  //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  //std::cout << "Hello from processor " << rank << "\n";
  //MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  Kokkos::finalize();

  return 0;
}