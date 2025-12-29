#pragma once

#include "core/Params.hpp"
#include "preprocess/FlowField.hpp" 

namespace orion::preprocess {

/**
 * @brief 初始化来流条件 (Init Inflow)
 * 对应 Fortran: subroutine init_inflow
 * * @param params 全局参数，将修改其中的 flow 成员
 * @param fs 流场数据，用于分配残差监控数组 (nres, res)
 */
void init_inflow(orion::core::Params& params, FlowFieldSet& fs);

} // namespace orion::preprocess