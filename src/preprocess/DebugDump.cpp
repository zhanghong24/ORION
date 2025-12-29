// src/preprocess/DebugDump.cpp
#include "preprocess/DebugDump.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

namespace orion::preprocess {

static inline void analyze_flag_face(const OrionArray<int>& a,
                                     long long& nnz,
                                     int& maxv)
{
  nnz = 0;
  maxv = 0;
  const int* p = a.hostPtr();
  const std::size_t n = a.size();
  for (std::size_t i = 0; i < n; ++i) {
    const int v = p[i];
    if (v != 0) {
      ++nnz;
      if (v > maxv) maxv = v;
    }
  }
}

void debug_dump_local_blocks_and_bcflag(const mesh::MultiBlockGrid& grid,
                                        const FlowFieldSet& fs)
{
  const int rank = orion::core::Runtime::rank();
  const int nprocs = orion::core::Runtime::size();

  // 为了避免并发输出乱序：按 rank 轮流打印
  for (int r = 0; r < nprocs; ++r) {
    orion::core::Runtime::barrier();

    if (r != rank) continue;

    std::ostringstream os;
    os << "================ [Rank " << rank << "] DebugDump ================\n";
    os << "local_blocks = " << fs.local_block_ids.size() << ", ng=" << fs.ng
       << ", nvar=" << fs.nvar << ", nprim=" << fs.nprim << "\n";

    // 面编号说明（与你 Fortran mb_flg(1..6) 对齐）
    os << "bc_flag faces: "
       << "1:i-min  2:i-max  3:j-min  4:j-max  5:k-min  6:k-max\n";

    for (std::size_t t = 0; t < fs.local_block_ids.size(); ++t) {
      const int nb0 = fs.local_block_ids[t];      // 0-based
      const int nb1 = nb0 + 1;                    // 1-based for printing

      const auto& bg = grid.blocks[(std::size_t)nb0];
      const auto& bf = fs.blocks[(std::size_t)nb0];

      os << "  [block " << nb1 << "] dim=("
         << bg.idim << "," << bg.jdim << "," << bg.kdim << ") ";

      if (!bf.allocated()) {
        os << " <NOT allocated>\n";
        continue;
      }
      os << "\n";

      // 逐面统计
      for (int face = 0; face < 6; ++face) {
        const auto& f = bf.bc_flag[(std::size_t)face];
        long long nnz = 0;
        int maxv = 0;
        analyze_flag_face(f, nnz, maxv);

        os << "    face " << (face + 1)
           << " size=" << f.size()
           << " nnz=" << nnz
           << " max_region=" << maxv
           << " has_nonzero=" << ((nnz > 0) ? "YES" : "NO")
           << "\n";
      }
    }

    os << "===============================================================\n";
    std::cout << os.str() << std::flush;

    orion::core::Runtime::barrier();
  }

  // 最后再同步一次
  orion::core::Runtime::barrier();
}

} // namespace orion::preprocess
