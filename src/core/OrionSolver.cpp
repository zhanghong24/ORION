#include "core/OrionSolver.hpp"
#include "core/Runtime.hpp"
#include "mesh/Plot3DReader.hpp"
#include "mesh/GridDistributor.hpp"
#include "mesh/MetricComputer.hpp"
#include "bc/BCReader.hpp"
#include "bc/BCDistributor.hpp"
#include "bc/BCPreprocess.hpp"
#include "bc/PhysicalBC.hpp"
#include "preprocess/Inflow.hpp"
#include "preprocess/InitialCondition.hpp"
#include "solver/StateUpdater.hpp"
#include "solver/TimeIntegrator.hpp"
#include "solver/FluxComputer.hpp"
#include "solver/InviscidFluxComputer.hpp"
#include "solver/ResidualMonitor.hpp"
#include "postprocess/PostProcess.hpp"

#include <iostream>
#include <iomanip>

namespace orion {

void OrionSolver::load_parameters(const std::string& filename) {
    if (core::Runtime::is_root()) {
        std::cout << "------------------------------------------------\n";
        std::cout << "[OrionSolver] Loading parameters from: " << filename << "\n";
    }
    params_.load_namelist_file(filename);
}

void OrionSolver::preprocess() {
    if (core::Runtime::is_root()) {
        std::cout << "[OrionSolver] Starting Preprocessing...\n";
        std::cout << "------------------------------------------------\n";
    }

    // 1. 根节点读取网格和边界条件
    if (core::Runtime::is_root()) {
        grid_ = mesh::Plot3DReader::read(params_.filename.gridname, params_.inflow.ndim);
        
        bc_ = bc::BCReader::read_parallel_bc_root_only(
                params_.filename.bcname, grid_, core::Runtime::nproc,
                {.mbsmarker = 1976, .verbose = false}
            );
    }

    // 2. 并行分发网格
    mesh::distrib_grid_fast(grid_, bc_.block_pid,
                            core::Runtime::myid,
                            core::Runtime::nproc,
                            {.master=0, .max_chunk_bytes=256ull*1024*1024, .verbose=false});

    // 3. 并行分发边界条件
    bc::distrib_bc_fast(bc_,
                        core::Runtime::myid,
                        core::Runtime::nproc,
                        {.master=0, .verbose=false});

    // 4. 设置边界索引
    bc::set_bc_index(bc_);

    // =========================================================
    // [NEW] 5. 计算边界拓扑映射 (复刻 Fortran analyze_bc)
    // =========================================================
    if (core::Runtime::is_root()) {
        std::cout << "[OrionSolver] Preparing BC Topology (analyze_bc)...\n";
    }
    bc::prepare_bc_topology(bc_);

    // 5. 分配流场内存 (Metrics, Q, Prim等)
    // 注意：这里 nvar=5, nprim=6 可以写死，也可以做到 params 里
    fs_ = preprocess::allocate_other_variable(grid_, bc_, params_,
                                              /*nvar=*/5,
                                              /*nprim=*/6);

    // 6. 填充边界标记
    preprocess::fill_bc_flag(grid_, bc_, fs_);
    
    // 7. 计算网格度量 (Metrics & Jacobian)
    mesh::compute_grid_metrics(grid_, fs_, params_);

    // 8. 检查网格质量 (负体积检查)
    mesh::check_grid_metrics(grid_, fs_, bc_);

    // 9. 初始化来流参数 (计算 roo, uoo, reynolds 等)
    preprocess::init_inflow(params_, fs_);

    // 10. 初始化流场初值 (Q & Prim)
    preprocess::initialize_flow(params_, fs_);

    // 11. 更新温度场 (热力学一致性)
    preprocess::update_temperature(params_, fs_);

    if (core::Runtime::is_root()) {
        std::cout << "------------------------------------------------\n";
        std::cout << "[OrionSolver] Preprocessing Finished Successfully.\n";
        std::cout << "------------------------------------------------\n";
    }
}

void OrionSolver::solve()
{
    const auto& ctrl = params_.control; // 简化引用

    // ---------------------------------------------------------
    // 1. 求解器启动信息报告
    // ---------------------------------------------------------
    if (core::Runtime::is_root()) {
        std::cout << "\n";
        std::cout << "=================================================================\n";
        std::cout << "                     ORION SOLVER STARTED                        \n";
        std::cout << "=================================================================\n";
        std::cout << "  Configuration:\n";
        std::cout << "    - Start Mode (nstart) : " << ctrl.nstart << (ctrl.nstart==0 ? " (Cold Start)" : " (Restart)") << "\n";
        std::cout << "    - Max Steps  (nomax)  : " << ctrl.nomax << "\n";
        std::cout << "    - Save Freq  (nplot)  : " << ctrl.nplot << "\n";
        std::cout << "    - Method     (method) : " << ctrl.method << "\n";
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "  Step      Physical Time        Residual(Avg)        Residual(Max)\n";
        std::cout << "-----------------------------------------------------------------\n" << std::flush;
    }

    // ---------------------------------------------------------
    // 2. 循环变量初始化
    // ---------------------------------------------------------
    int step = 0;          // 当前时间步

    double start_wtime = MPI_Wtime();
    double elapsed_time = 0.0;
    
    // 如果是重启动，step 和 phys_time 应该从文件读取 (目前暂定为0)
    // if (ctrl.nstart != 0) { ... }
    
    // ---------------------------------------------------------
    // 3. 主时间步进循环 (Main Loop)
    // ---------------------------------------------------------
    
    while (step < ctrl.nomax) {

        halo_exchanger_.exchange_bc(bc_, fs_);
        
        orion::bc::apply_physical_bc(bc_, fs_, params_);

        solver::StateUpdater::update_flow_states(fs_, params_);

        solver::TimeIntegrator::calculate_time_step(fs_, params_);

        if (params_.flowtype.nvis == 1)
        {
            solver::FluxComputer::compute_viscous_rhs(fs_, params_);

            halo_exchanger_.average_interface_residuals(bc_, fs_);
        }

        solver::InviscidFluxComputer::compute_inviscid_rhs(fs_, params_);

        halo_exchanger_.average_interface_residuals(bc_, fs_);

        solver::StateUpdater::update_solution(fs_, bc_, params_);

        auto res_stats = solver::ResidualMonitor::compute(fs_, params_);

        step++;
        elapsed_time = MPI_Wtime() - start_wtime;

        if (core::Runtime::is_root() && step % 10 == 0) {
             std::cout << "  " << std::setw(2) << step 
                       << "  Time: " << std::scientific << std::setprecision(4) << elapsed_time << "s"
                       << "  RMS: " << res_stats.rms_residual 
                       << "  Max: " << res_stats.max_residual 
                       << " @" << res_stats.max_loc[0] << "," 
                       << res_stats.max_loc[1] << "," << res_stats.max_loc[2] << "," << res_stats.max_loc[3]
                       << std::endl;
        }
    }

    postprocess::PostProcess::write_solution(fs_, grid_, params_, step);

    // ---------------------------------------------------------
    // 4. 求解结束
    // ---------------------------------------------------------
    if (core::Runtime::is_root()) {
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "  Calculation Finished at Step " << step << "\n";
        std::cout << "=================================================================\n" << std::flush;
    }
}

} // namespace orion