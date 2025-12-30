#pragma once
#include <array>
#include <string>
#include <vector>

namespace orion::bc {

struct BCRegion {
  int bctype = 0;
  int nbs = -1;   // source block id (1-based)
  int nbt = -1;   // target block id (only when bctype < 0)
  int ibcwin = 0; // target window id (only when bctype < 0)

  std::array<int,3> s_st{{0,0,0}};
  std::array<int,3> s_ed{{0,0,0}};
  std::array<int,3> t_st{{0,0,0}};
  std::array<int,3> t_ed{{0,0,0}};
  
  int s_nd = 0; // Normal dir: 1=i, 2=j, 3=k
  int s_lr = 0; // Face: -1=Min, 1=Max
  int s_fix = 0; // Fixed index
  std::array<int,3> s_lr3d{{0,0,0}}; // 3D direction vector

  int t_nd = 0;
  int t_lr = 0;
  int t_fix = 0;
  std::array<int,3> t_lr3d{{0,0,0}};

  // --- NEW: Communication Buffers & Topology (Replicating Fortran) ---
  
  // Received data buffer (mimics Fortran qpvpack)
  // Layout: Flat array. Fortran is (i, j, k, var). 
  // We will store as: [i][j][k][var] flattened.
  std::vector<double> qpvpack; 
  
  // Dimensions of the received buffer (Source window size)
  std::array<int,3> pack_dims{{0,0,0}}; 

  // Topology Mapping (mimics Fortran image, jmage, kmage)
  // These map the Local Ghost (i,j,k) -> Index in qpvpack (it, jt, kt)
  // Dimensions match the Local Ghost window size.
  std::vector<int> image;
  std::vector<int> jmage;
  std::vector<int> kmage;

  // Dimensions of the local ghost window (for indexing image/jmage/kmage)
  std::array<int,3> map_dims{{0,0,0}};

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
