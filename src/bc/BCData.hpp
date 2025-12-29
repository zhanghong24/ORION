#pragma once
#include <array>
#include <string>
#include <vector>

namespace orion::bc {

struct BCRegion {
  int bctype = 0;
  int nbs = -1;   // source block id (1-based, to match legacy Fortran)
  int nbt = -1;   // target block id (only when bctype < 0)
  int ibcwin = 0; // window id / orientation flag (only when bctype < 0)

  std::array<int,3> s_st{{0,0,0}};
  std::array<int,3> s_ed{{0,0,0}};
  std::array<int,3> t_st{{0,0,0}};
  std::array<int,3> t_ed{{0,0,0}};
  int s_nd = 0; // 法向方向: 1=i, 2=j, 3=k
  int s_lr = 0; // 位置标识: -1=Min面(start=1), 1=Max面(start=dim)

  bool is_connect() const { return bctype < 0; }
};

struct BlockBC {
  std::string blockname;
  int nregions = 0;
  std::vector<BCRegion> regions;
  // NEW: BC traversal order (1-based region indices like Fortran)
  std::vector<int> bcindexs;
};

struct BCData {
  int flow_solver_id = 0;
  int nprocs_in_file = 0;
  int number_of_blocks = 0;

  // 1-based indexing semantics in file; we store vectors sized nblocks
  std::vector<int> block_pid;        // mb_pids(nb)
  std::vector<BlockBC> block_bc;     // mb_bc(nb)
};

} // namespace orion::bc
