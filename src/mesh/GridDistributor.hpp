#pragma once
#include "mesh/MultiBlockGrid.hpp"
#include <vector>
#include <cstddef>

namespace orion::mesh {

struct GridDistribOptions {
  int master = 0;
  std::size_t max_chunk_bytes = 256ull * 1024ull * 1024ull; // 256MB per message
  bool verbose = true;
};

void distrib_grid_fast(MultiBlockGrid& grid,
                       std::vector<int>& block_pid_1based,
                       int myrank, int nranks,
                       const GridDistribOptions& opt = {});

} // namespace orion::mesh
