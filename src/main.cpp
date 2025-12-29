#include "core/Runtime.hpp"
#include "core/OrionSolver.hpp"
#include <iostream>

int main(int argc, char** argv)
{
    // 1. MPI 环境初始化
    orion::core::Runtime::init(argc, argv);

    try {
        // 2. 实例化求解器
        orion::OrionSolver solver;

        // 3. 加载参数
        solver.load_parameters("param.dat");

        // 4. 执行预处理流程
        solver.preprocess();

        // 5. 执行计算
        solver.solve(); 

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        orion::core::Runtime::finalize(); // 确保 MPI 正确退出
        return -1;
    }

    // 6. 清理退出
    orion::core::Runtime::finalize();
    return 0;
}