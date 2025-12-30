// src/preprocess/FlowField.cpp
#include "preprocess/FlowField.hpp"

#include <algorithm>
#include <iostream>

namespace orion::preprocess {

int ghost_layers_from_scheme(int nscheme)
{
  // 你要求：接口先保留，先明确 41 => 3
  // 后面你把其它 scheme 对应 ng 补上即可
  if (nscheme == 41) return 3; // WCNS-E-5
  // 保守默认（你也可以先设 2 或 1，看你后面要不要兼容）
  return 3;
}

static inline bool is_local_block(const bc::BCData& bc, int nb0 /*0-based*/)
{
  // bc.block_pid 是 1-based pid（Fortran pid），Runtime::rank 是 0-based
  const int mypid1 = orion::core::Runtime::rank() + 1;
  return bc.block_pid[(std::size_t)nb0] == mypid1;
}

FlowFieldSet allocate_other_variable(const mesh::MultiBlockGrid& grid,
                                     const bc::BCData& bc,
                                     const core::Params& params,
                                     int nvar,
                                     int nprim)
{
  if (grid.nblocks <= 0) throw std::runtime_error("allocate_other_variable: grid.nblocks <= 0");
  if ((int)grid.blocks.size() != grid.nblocks)
    throw std::runtime_error("allocate_other_variable: grid.blocks size mismatch");
  if (bc.number_of_blocks != grid.nblocks)
    throw std::runtime_error("allocate_other_variable: bc.number_of_blocks != grid.nblocks");
  if ((int)bc.block_pid.size() != grid.nblocks)
    throw std::runtime_error("allocate_other_variable: bc.block_pid size mismatch");

  FlowFieldSet fs;
  fs.ng = ghost_layers_from_scheme(params.technic.nscheme);
  fs.nvar = nvar;
  fs.nprim = nprim;
  fs.blocks.resize((std::size_t)grid.nblocks);

  fs.local_block_ids.clear();
  fs.local_block_ids.reserve((std::size_t)grid.nblocks);

  const int ng = fs.ng;

  for (int nb = 0; nb < grid.nblocks; ++nb) {
    if (!is_local_block(bc, nb)) {
      // 非本 rank 的块：不分配（复刻 Fortran）
      continue;
    }

    fs.local_block_ids.push_back(nb);

    const int idim = grid.blocks[(std::size_t)nb].idim;
    const int jdim = grid.blocks[(std::size_t)nb].jdim;
    const int kdim = grid.blocks[(std::size_t)nb].kdim;

    if (idim <= 0 || jdim <= 0 || kdim <= 0) {
      throw std::runtime_error("allocate_other_variable: invalid block dims");
    }

    // 统一 ghost：分配 (idim+2ng, jdim+2ng, kdim+2ng)
    const int nx = idim + 2 * ng;
    const int ny = jdim + 2 * ng;
    const int nz = kdim + 2 * ng;

    auto& bf = fs.blocks[(std::size_t)nb];

    bf.vol.resize(nx, ny, nz);
    bf.metrics.resize(nx, ny, nz, 9);
    bf.q.resize(nx, ny, nz, fs.nvar);
    bf.dq.resize(nx, ny, nz, fs.nvar);
    bf.prim.resize(nx, ny, nz, fs.nprim);
    bf.c.resize(nx, ny, nz);
    bf.mu.resize(nx, ny, nz);
    bf.dt.resize(nx,ny,nz);
    
    // [修复] 谱半径需要存储 3 个方向的分量 (xi, eta, zeta)
    // 之前分配为 .resize(nx,ny,nz) 导致越界
    bf.spec_radius.resize(nx, ny, nz, 3);
    bf.spec_radius_visc.resize(nx, ny, nz, 3);

    // bc_flag 面阵列：只存“真实边界面”上的点（不带 ghost）
    // 与 Fortran mb_flg(nb,1..6) 的尺寸一致
    // 0:i=1 -> (jdim,kdim)
    // 1:i=idim -> (jdim,kdim)
    // 2:j=1 -> (idim,kdim)
    // 3:j=jdim -> (idim,kdim)
    // 4:k=1 -> (idim,jdim)
    // 5:k=kdim -> (idim,jdim)
    bf.bc_flag[0].resize(jdim, kdim);
    bf.bc_flag[1].resize(jdim, kdim);
    bf.bc_flag[2].resize(idim, kdim);
    bf.bc_flag[3].resize(idim, kdim);
    bf.bc_flag[4].resize(idim, jdim);
    bf.bc_flag[5].resize(idim, jdim);

    // 初始化为 0（表示未标记 region）
    for (auto& f : bf.bc_flag) f.fill(0);
  }

  return fs;
}

static inline int face_id_from_window(const mesh::BlockGrid& bg,
                                      const std::array<int,3>& s_st,
                                      const std::array<int,3>& s_ed)
{
  // bc 文件是 1-based index；bg.idim/jdim/kdim 是维度
  // 判断哪个方向常数（面），再判断是 min 还是 max 面
  const int idim = bg.idim;
  const int jdim = bg.jdim;
  const int kdim = bg.kdim;

  if (s_st[0] == s_ed[0]) {
    const int i = s_st[0];
    if (i == 1)   return 0;
    if (i == idim) return 1;
    throw std::runtime_error("BC window is i-constant but not on i=1/idim");
  }
  if (s_st[1] == s_ed[1]) {
    const int j = s_st[1];
    if (j == 1)   return 2;
    if (j == jdim) return 3;
    throw std::runtime_error("BC window is j-constant but not on j=1/jdim");
  }
  if (s_st[2] == s_ed[2]) {
    const int k = s_st[2];
    if (k == 1)   return 4;
    if (k == kdim) return 5;
    throw std::runtime_error("BC window is k-constant but not on k=1/kdim");
  }

  throw std::runtime_error("BC window is not a face (no constant direction)");
}

void fill_bc_flag(const mesh::MultiBlockGrid& grid,
                  const bc::BCData& bc,
                  FlowFieldSet& fs)
{
  if ((int)fs.blocks.size() != grid.nblocks)
    throw std::runtime_error("fill_bc_flag: fs.blocks size mismatch");
  if (bc.number_of_blocks != grid.nblocks)
    throw std::runtime_error("fill_bc_flag: bc.number_of_blocks mismatch");

  for (int nb = 0; nb < grid.nblocks; ++nb) {
    if (!is_local_block(bc, nb)) continue; // 只处理本 rank 的块
    auto& bf = fs.blocks[(std::size_t)nb];
    if (!bf.allocated()) continue;

    const auto& bg  = grid.blocks[(std::size_t)nb];
    const auto& bcb = bc.block_bc[(std::size_t)nb];

    // 遵循你 Fortran 的逻辑：按 bcindexs 顺序遍历
    // bcindexs 是 1-based region id
    for (int ib = 0; ib < (int)bcb.bcindexs.size(); ++ib) {
      const int nr1 = bcb.bcindexs[(std::size_t)ib]; // 1..nregions
      const int nr0 = nr1 - 1;
      if (nr0 < 0 || nr0 >= (int)bcb.regions.size())
        throw std::runtime_error("fill_bc_flag: bcindex out of range");

      const auto& reg = bcb.regions[(std::size_t)nr0];
      const auto s_st = reg.s_st;
      const auto s_ed = reg.s_ed;

      const int face = face_id_from_window(bg, s_st, s_ed);

      // 将 1-based (i,j,k) 转成 0-based 用于 OrionArray
      const int i0 = s_st[0] - 1;
      const int i1 = s_ed[0] - 1;
      const int j0 = s_st[1] - 1;
      const int j1 = s_ed[1] - 1;
      const int k0 = s_st[2] - 1;
      const int k1 = s_ed[2] - 1;

      // 注意：在 face 上写入 region 编号 nr1（保持与 Fortran 一致）
      if (face == 0 || face == 1) {
        // i-face => (j,k)
        for (int k = k0; k <= k1; ++k)
          for (int j = j0; j <= j1; ++j)
            bf.bc_flag[(std::size_t)face](j, k) = nr1;
      } else if (face == 2 || face == 3) {
        // j-face => (i,k)
        for (int k = k0; k <= k1; ++k)
          for (int i = i0; i <= i1; ++i)
            bf.bc_flag[(std::size_t)face](i, k) = nr1;
      } else {
        // k-face => (i,j)
        for (int j = j0; j <= j1; ++j)
          for (int i = i0; i <= i1; ++i)
            bf.bc_flag[(std::size_t)face](i, j) = nr1;
      }
    }
  }
}

} // namespace orion::preprocess
