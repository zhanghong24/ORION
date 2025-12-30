// src/preprocess/FlowField.hpp
#pragma once

#include "core/OrionArray.hpp"
#include "core/Params.hpp"
#include "core/Runtime.hpp"
#include "mesh/MultiBlockGrid.hpp"
#include "bc/BCData.hpp"

#include <array>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <string>

namespace orion::preprocess {

// 守恒量 q,dq 统一；原始量 prim 统一（rho,u,v,w,p,t...）
struct BlockField {
  // geometry / metrics 
  OrionArray<double> vol;   // (idim+2ng, jdim+2ng, kdim+2ng)

  // 0:kcx, 1:kcy, 2:kcz (xi gradients)
  // 3:etx, 4:ety, 5:etz (eta gradients)
  // 6:ctx, 7:cty, 8:ctz (zeta gradients)
  OrionArray<double> metrics; // (idim+2ng, jdim+2ng, kdim+2ng, 9)

  // flow variables
  OrionArray<double> q;     // (idim+2ng, jdim+2ng, kdim+2ng, nvar)
  OrionArray<double> dq;    // same as q
  OrionArray<double> prim;  // (idim+2ng, jdim+2ng, kdim+2ng, nprim)

  OrionArray<double> dt;

  OrionArray<double> c;     // Speed of Sound (idim+2ng, jdim+2ng, kdim+2ng)
  OrionArray<double> mu;    // Viscosity (Laminar+Turbulent?) (idim+2ng, jdim+2ng, kdim+2ng)

  // 谱半径 (Spectral Radius)
  // 0: xi (sra), 1: eta (srb), 2: zeta (src)
  OrionArray<double> spec_radius; 
  
  // [新增] 粘性谱半径 (Viscous Spectral Radius)
  // 0: xi (srva), 1: eta (srvb), 2: zeta (srvc)
  OrionArray<double> spec_radius_visc;
  
  // boundary region flag on 6 faces
  // face order (与 Fortran mb_flg(nb,1..6) 对齐):
  // 0: i=1 plane
  // 1: i=idim plane
  // 2: j=1 plane
  // 3: j=jdim plane
  // 4: k=1 plane
  // 5: k=kdim plane
  std::array<OrionArray<int>, 6> bc_flag;

  void clear() {
    vol.clear();
    metrics.clear();
    q.clear();
    dq.clear();
    prim.clear();
    c.clear();
    mu.clear();
    dt.clear();
    spec_radius.clear();
    spec_radius_visc.clear();
    for (auto& f : bc_flag) f.clear();
  }

  bool allocated() const noexcept { return vol.size() != 0; }
};

struct FlowFieldSet {
  int ng = 0;
  int nvar = 0;
  int nprim = 0;

  // 全局 nblocks 尺寸，但只有本 rank 的块会真正分配
  std::vector<BlockField> blocks;

  // 本 rank 拥有的 block 列表（0-based block id）
  std::vector<int> local_block_ids;

  // residual 
  std::vector<double> residual_norm;
  std::vector<double> residual_max;
};

// --- public API ---
int ghost_layers_from_scheme(int nscheme);

// 只对本 rank 的 block 分配必要数组
FlowFieldSet allocate_other_variable(const mesh::MultiBlockGrid& grid,
                                     const bc::BCData& bc,
                                     const core::Params& params,
                                     int nvar,
                                     int nprim);

// 填充 bc_flag：把每个 region 编号写到对应 face 的窗口内
// 注意：bcindexs 是 1-based region index（你前面 set_bc_index 已按 Fortran 做了）
void fill_bc_flag(const mesh::MultiBlockGrid& grid,
                  const bc::BCData& bc,
                  FlowFieldSet& fs);

} // namespace orion::preprocess
