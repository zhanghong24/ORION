#pragma once
#include "preprocess/FlowField.hpp"
#include "core/Params.hpp"

namespace orion::solver {

class TimeIntegrator {
public:
    // 主入口：对应 subroutine timestep_tgh
    static void calculate_time_step(orion::preprocess::FlowFieldSet& fs, 
                                    orion::core::Params& params);

private:
    // 子函数 1：对应 subroutine spectrum_tgh
    // [修复] 增加 ng 参数
    static void calculate_spectrum(orion::preprocess::BlockField& bf, 
                                   const orion::core::Params& params,
                                   int ng);

    // 子函数 2：对应 subroutine localdt0
    // [修复] 增加 ng 参数
    static void calculate_local_dt_cell(orion::preprocess::BlockField& bf, 
                                        const orion::core::Params& params,
                                        int ng);
    
    // 辅助：全局时间步长归约 (MPI AllReduce)
    static double reduce_global_dt(double local_min_dt);
};

} // namespace orion::solver
