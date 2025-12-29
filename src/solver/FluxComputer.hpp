#pragma once
#include "preprocess/FlowField.hpp"
#include "core/Params.hpp"
#include "core/OrionArray.hpp"

namespace orion::solver {

class FluxComputer {
public:
    /**
     * @brief 计算粘性残差 (Viscous RHS)
     * 对应 Fortran: subroutine r_h_s_vis
     * 只有当 params.flowtype.nvis == 1 时才会执行实际计算。
     */
    static void compute_viscous_rhs(orion::preprocess::FlowFieldSet& fs, 
                                    const orion::core::Params& params);

private:
    /**
     * @brief 单个 Block 的粘性通量计算核心
     * 对应 Fortran: subroutine WCNSE5_VIS_VIRTUAL
     * @param bf 当前块的数据
     * @param params 全局参数
     * @param ng Ghost 层数
     */
    static void compute_viscous_flux_block(orion::preprocess::BlockField& bf,
                                           const orion::core::Params& params,
                                           int ng);

    // -----------------------------------------------------------
    // 导数计算子程序
    // -----------------------------------------------------------
    
    // 计算网格点上的导数 (Cell Center Derivatives)
    // 对应 Fortran: UVWT_DER_4th_virtual
    static void compute_derivatives_cell(const orion::preprocess::BlockField& bf, 
                                         orion::OrionArray<double>& duvwt, 
                                         int ng);

    // 计算半点(界面)上的导数 (Interface Derivatives)
    // 对应 Fortran: UVWT_DER_4th_half_virtual
    static void compute_derivatives_face(const orion::preprocess::BlockField& bf, 
                                         orion::OrionArray<double>& duvwt_mid, 
                                         int ng);
};

} // namespace orion::solver
