#pragma once
#include <cstddef>
#include <vector>
#include <string>
#include "core/OrionArray.hpp"

namespace orion::mesh {

struct BlockGrid {
  int idim = 0, jdim = 0, kdim = 0;

  // coordinates
  orion::OrionArray<double> x;
  orion::OrionArray<double> y;
  orion::OrionArray<double> z;

  std::size_t npoints() const {
    return static_cast<std::size_t>(idim) * jdim * kdim;
  }
};

struct MultiBlockGrid {
  int ndim = 3;
  int nblocks = 0;
  int nmax = 0;
  std::size_t total_points = 0;

  std::vector<BlockGrid> blocks;

  void clear() {
    ndim = 3;
    nblocks = 0;
    nmax = 0;
    total_points = 0;
    blocks.clear();
  }
};

} // namespace orion::mesh