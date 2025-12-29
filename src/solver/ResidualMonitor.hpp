#pragma once
#include "preprocess/FlowField.hpp"
#include "core/Params.hpp"
#include <vector>

namespace orion::solver {

struct ResidualStats {
    double rms_residual = 0.0; // L2 Norm
    double max_residual = 0.0; // L-inf Norm
    int max_loc[5] = {0};      // [nb_global, i, j, k, var]
};

class ResidualMonitor {
public:
    /**
     * @brief 计算全场残差统计信息
     * 对应 Fortran: residual_curve
     */
    static ResidualStats compute(const orion::preprocess::FlowFieldSet& fs, 
                                 const orion::core::Params& params);
};

} // namespace orion::solver