#pragma once
#include "bc/BCData.hpp"

namespace orion::bc {

struct BCDistribOptions {
  int  master  = 0;
  bool verbose = false;
};

/// Fast distribution of BC topology:
/// - root serializes the whole BCData into a byte buffer
/// - MPI_Bcast size + MPI_Bcast buffer
/// - all ranks reconstruct BCData
void distrib_bc_fast(BCData& bc,
                     int myrank, int nranks,
                     const BCDistribOptions& opt = {});

} // namespace orion::bc
