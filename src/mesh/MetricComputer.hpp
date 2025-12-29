#pragma once

#include "mesh/MultiBlockGrid.hpp"
#include "preprocess/FlowField.hpp"
#include "core/Params.hpp"

namespace orion::mesh {

/**
 * @brief Computes grid metrics (Jacobians) and volume.
 * Corresponds to Fortran 'set_grid_derivative' and 'GRID_DERIVATIVE_gcl'.
 * * Logic:
 * 1. Checks grid thickness (must be > 2*ng).
 * 2. Calculates derivatives of x,y,z using high-order schemes.
 * 3. Computes metrics (kcx, etz, etc.) via cross-products.
 * 4. Applies Visbal's GCL correction if enabled.
 * * Results are stored in fs.blocks[nb].metrics and fs.blocks[nb].vol.
 * Note: Only computes for local blocks owned by this rank.
 */
void compute_grid_metrics(const MultiBlockGrid& grid,
                          orion::preprocess::FlowFieldSet& fs,
                          const orion::core::Params& params);

/**
 * @brief 检查网格体积（Jacobian）。
 * 对应 Fortran 'check_grid_derivative'
 * * 1. 对特定边界类型 (71,72,73) 强制修正体积为 pole_vol。
 * 2. 统计最小/最大体积。
 * 3. 检查是否存在负体积 (< sml_vol)，若有则修正并报错。
 * 4. 检查是否退化为 2D 网格。
 */
void check_grid_metrics(const MultiBlockGrid& grid,
                        orion::preprocess::FlowFieldSet& fs,
                        const orion::bc::BCData& bc);

};