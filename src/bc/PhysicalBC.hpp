#pragma once

#include "bc/BCData.hpp"
#include "preprocess/FlowField.hpp"
#include "core/Params.hpp"

namespace orion::bc {

/**
 * @brief 应用物理边界条件 (对应 Fortran: boundary_sequence 中 bctype > 0 的部分)
 * * 遍历所有本地块的边界区域，对于 bctype > 0 的区域，调用对应的物理处理函数。
 * 这一步通常在 Halo Exchange 之前或之后调用，用于填充物理边界的 Ghost Cells。
 * * @param bc     边界定义数据
 * @param fs     流场数据 (将修改 bf.prim 的 ghost 区域)
 * @param params 全局参数 (需要 inflow 参数如 uoo, voo, twall 等)
 */
void apply_physical_bc(const BCData& bc, 
                       orion::preprocess::FlowFieldSet& fs,
                       const orion::core::Params& params);

} // namespace orion::bc