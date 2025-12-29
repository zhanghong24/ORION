// src/core/Runtime.cpp
#include "core/Runtime.hpp"

#include <mpi.h>
#include <iostream>

namespace orion::core {

// static member definitions
int Runtime::myid  = -1;
int Runtime::nproc = 0;

void Runtime::init(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

#ifdef ORION_USE_CPU
    const char* backend = "CPU";
#elif defined(ORION_USE_CUDA)
    const char* backend = "CUDA";
#elif defined(ORION_USE_HIP)
    const char* backend = "HIP";
#elif defined(ORION_USE_DCU)
    const char* backend = "DCU";
#else
    const char* backend = "UNKNOWN";
#endif

  if (is_root()) {
    std::cout << "====================================\n";
    std::cout << " ORION Runtime Initialized\n";
    std::cout << " Backend: " << backend << "\n";
    std::cout << " MPI ranks: " << nproc << "\n";
    std::cout << "====================================\n";
  }
}

void Runtime::finalize()
{
  if (is_root()) {
    std::cout << "ORION Runtime Finalized\n";
  }
  MPI_Finalize();
}

void Runtime::barrier()
{
  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace orion
