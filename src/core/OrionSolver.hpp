#pragma once

#include <string>
#include "core/Params.hpp"
#include "mesh/MultiBlockGrid.hpp"
#include "bc/BCData.hpp"
#include "preprocess/FlowField.hpp"
#include "solver/HaloExchanger.hpp"

namespace orion {

class OrionSolver {
public:
    OrionSolver() = default;
    ~OrionSolver() = default;

    // 1. 加载参数文件
    void load_parameters(const std::string& filename);

    // 2. 执行预处理 (读取网格、分发、计算Metrics、初始化流场)
    void preprocess();

    // 3. 执行时间步进
    void solve(); 

private:
    // --- 核心数据成员 ---
    core::Params params_;
    mesh::MultiBlockGrid grid_;
    bc::BCData bc_;
    preprocess::FlowFieldSet fs_; // fields
    solver::HaloExchanger halo_exchanger_;
};

} // namespace orion