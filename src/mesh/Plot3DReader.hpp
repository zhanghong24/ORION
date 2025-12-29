#pragma once
#include <string>
#include "mesh/MultiBlockGrid.hpp"

namespace orion::mesh {

struct Plot3DReaderOptions {
  bool try_ascii_fallback = true;
  bool verbose = false;
};

class Plot3DReader {
public:
  static MultiBlockGrid read(const std::string& filename, int ndim,
                             const Plot3DReaderOptions& opt = {});
};

} // namespace orion::mesh
