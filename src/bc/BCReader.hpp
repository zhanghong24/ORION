#pragma once
#include <string>
#include "bc/BCData.hpp"
#include "mesh/MultiBlockGrid.hpp"

namespace orion::bc {

struct BCReaderOptions {
  int mbsmarker = 1976;     // 示例文件第一行就是 1976；通过参数覆盖
  bool verbose = true;
};

class BCReader {
public:
  // numprocs: 当前 MPI size (Runtime::nproc)
  // grid: 已经读入的多块网格（用于校验维度 nblocks 和每块尺寸）
  static BCData read_parallel_bc_root_only(const std::string& bcname,
                                           const orion::mesh::MultiBlockGrid& grid,
                                           int numprocs,
                                           const BCReaderOptions& opt = {});
};

} // namespace orion::bc
