#pragma once
#include "preprocess/FlowField.hpp"
#include "core/Params.hpp"
#include "core/OrionArray.hpp"
#include "bc/BCData.hpp" // [新增] 需要这个头文件

namespace orion::solver {

class StateUpdater {
public:
    /**
     * @brief 更新辅助流场变量 (c, T, mu, Q)
     */
    static void update_flow_states(orion::preprocess::FlowFieldSet& fs, 
                                   const orion::core::Params& params);

    /**
     * @brief 隐式时间推进求解与更新 (LU-SGS + Update)
     * [修正] 增加了 const orion::bc::BCData& bc 参数
     */
    static void update_solution(orion::preprocess::FlowFieldSet& fs, 
                                const orion::bc::BCData& bc,
                                const orion::core::Params& params);
};

} // namespace orion::solver