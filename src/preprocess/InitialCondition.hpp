#pragma once

#include "core/Params.hpp"
#include "preprocess/FlowField.hpp"

namespace orion::preprocess {

/**
 * @brief 初始化流场 (Initialization)
 * 对应 Fortran: subroutine initialization & init_nstart0
 * * 根据 params.control.nstart 的值决定启动方式：
 * - nstart == 0: 冷启动 (Cold Start)，使用 params.inflow 中的无穷远参数均匀初始化全场。
 * - nstart != 0: 重启动 (Restart)，目前留空待实现。
 * * @param params 全局参数
 * @param fs 流场数据 (将被填充初值)
 */
void initialize_flow(const orion::core::Params& params, FlowFieldSet& fs);

/**
 * @brief 根据状态方程更新温度场 (T = Gamma * M^2 * P / Rho)
 * 对应 Fortran: subroutine initial_t
 * * 仅当 params.flowtype.nvis != 0 时执行。
 * 用于保证初始场或重启动场的热力学一致性，特别是对于粘性计算。
 * * @param params 全局参数 (读取 flowtype.nvis, inflow.gama, inflow.moo)
 * @param fs 流场数据 (读取 prim[0], prim[4] -> 更新 prim[5])
 */
void update_temperature(const orion::core::Params& params, FlowFieldSet& fs);

} // namespace orion::preprocess