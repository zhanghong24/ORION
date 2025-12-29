#pragma once
#include "preprocess/FlowField.hpp"
#include "core/Params.hpp"
#include "core/OrionArray.hpp"
#include <vector>

namespace orion::solver {

class InviscidFluxComputer {
public:
    /**
     * @brief 计算无粘残差 (Inviscid RHS)
     * 对应 Fortran: subroutine r_h_s_invis -> invcode -> inviscd3d
     * 假设 nflux = 3 (Roe), 忽略激波探测逻辑。
     * 结果将累加到 bf.dq 中 (dq += rhs)。
     */
    static void compute_inviscid_rhs(orion::preprocess::FlowFieldSet& fs, 
                                     const orion::core::Params& params);

private:
    /**
     * @brief 单个 Block 的无粘通量计算
     * 对应 Fortran: inviscd3d 的核心循环 (维数分裂)
     */
    static void compute_block_inviscid(orion::preprocess::BlockField& bf,
                                       const orion::core::Params& params,
                                       int ng);

    /**
     * @brief 1D 线通量计算 (Flux Line)
     * 对应 Fortran: WCNS_E_5_41 + flux_Roe
     * @param ni 当前线上的内部点数 (idim/jdim/kdim)
     * @param q_line 1D 原始变量数组 (含 Ghost, 6 vars: rho,u,v,w,p,T)
     * @param met_line 1D Metrics 数组 (5 vars: kx,ky,kz,kt,vol)
     * @param fc 输出的通量导数 (Flux Derivative)
     */
    static void compute_flux_line(int ni, 
                                  const std::vector<double>& q_line,
                                  const std::vector<double>& met_line,
                                  std::vector<double>& fc,
                                  const orion::core::Params& params);
};

} // namespace orion::solver