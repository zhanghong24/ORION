#include "core/Runtime.hpp"
#include "core/Params.hpp"
#include "mesh/Plot3DReader.hpp"
#include "mesh/GridDistributor.hpp"
#include "mesh/MetricComputer.hpp"
#include "bc/BCReader.hpp"
#include "bc/BCDistributor.hpp"
#include "bc/BCPreprocess.hpp"
#include "bc/PhysicalBC.hpp" 
#include "preprocess/FlowField.hpp"
#include "preprocess/InitialCondition.hpp"
#include "solver/HaloExchanger.hpp"
#include "solver/InviscidFluxComputer.hpp"
#include "solver/StateUpdater.hpp" 
#include "postprocess/PostProcess.hpp"

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <functional>
#include <iomanip>

// ---------------------------------------------------------------------------
// 辅助宏
// ---------------------------------------------------------------------------
#define LOG_INFO(msg) if(orion::core::Runtime::is_root()) std::cout << "[BENCH] " << msg << std::endl
#define LOG_PASS(msg) if(orion::core::Runtime::is_root()) std::cout << "\033[1;32m[PASS] " << msg << "\033[0m" << std::endl
#define LOG_FAIL(msg) if(orion::core::Runtime::is_root()) std::cout << "\033[1;31m[FAIL] " << msg << "\033[0m" << std::endl

// ---------------------------------------------------------------------------
// 测试 1: 通信指纹测试
// ---------------------------------------------------------------------------
void test_halo_exchange(orion::solver::HaloExchanger& exchanger,
                        orion::bc::BCData& bc, 
                        orion::preprocess::FlowFieldSet& fs) 
{
    LOG_INFO(">>> TEST 1: Halo Exchange Fingerprint Check");
    int my_rank = orion::core::Runtime::myid;

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        std::fill(bf.prim.hostPtr(), bf.prim.hostPtr() + bf.prim.size(), -999.0);

        int ni = bf.prim.dims()[0];
        int nj = bf.prim.dims()[1];
        int nk = bf.prim.dims()[2];

        for (int k = fs.ng; k < nk-fs.ng; ++k) {
            for (int j = fs.ng; j < nj-fs.ng; ++j) {
                for (int i = fs.ng; i < ni-fs.ng; ++i) {
                    bf.prim(i, j, k, 0) = (double)(nb + 1);
                    bf.prim(i, j, k, 1) = (double)my_rank;
                    bf.prim(i, j, k, 2) = (double)i;
                    bf.prim(i, j, k, 3) = (double)j;
                    bf.prim(i, j, k, 4) = (double)k;
                }
            }
        }
    }

    exchanger.exchange_bc(bc, fs);
    MPI_Barrier(MPI_COMM_WORLD);

    int local_err = 0;
    for (int nb : fs.local_block_ids) {
        auto& bcb = bc.block_bc[nb];
        auto& bf = fs.blocks[nb];
        for (const auto& reg : bcb.regions) {
            if (reg.nbt <= 0) continue;

            int ic = (std::abs(reg.s_st[0]) + std::abs(reg.s_ed[0])) / 2;
            int jc = (std::abs(reg.s_st[1]) + std::abs(reg.s_ed[1])) / 2;
            int kc = (std::abs(reg.s_st[2]) + std::abs(reg.s_ed[2])) / 2;

            int i_p = ic - 1 + fs.ng;
            int j_p = jc - 1 + fs.ng;
            int k_p = kc - 1 + fs.ng;

            int dir = reg.s_nd - 1;
            int shift = (reg.s_lr == 1) ? 1 : -1;
            if(dir==0) i_p += shift;
            if(dir==1) j_p += shift;
            if(dir==2) k_p += shift;

            double val = bf.prim(i_p, j_p, k_p, 0);
            if (std::abs(val - (double)reg.nbt) > 0.1) local_err++;
        }
    }

    int global_err = 0;
    MPI_Allreduce(&local_err, &global_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_err == 0) LOG_PASS("Halo Exchange Topology Matches Perfectly.");
    else LOG_FAIL("Halo Exchange Mismatch Found!");
}

// ---------------------------------------------------------------------------
// 测试 2: 几何度量测试 (真·立方体)
// ---------------------------------------------------------------------------
void test_metrics(orion::mesh::MultiBlockGrid& grid, 
                  orion::preprocess::FlowFieldSet& fs,
                  const orion::core::Params& params) 
{
    LOG_INFO(">>> TEST 2: Metric Consistency Check (Synthetic Unit Cube)");
    int err_count = 0;

    // [FIX] 必须遍历所有本地 Block，防止 Test 3 撞上未修改的 Block
    for (int nb : fs.local_block_ids) {
        auto& grid_blk = grid.blocks[nb];
        
        int ni = grid_blk.idim;
        int nj = grid_blk.jdim;
        int nk = grid_blk.kdim;

        // 强制改为完美笛卡尔网格 (x=i, y=j, z=k)
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    grid_blk.x(i, j, k) = (double)i;
                    grid_blk.y(i, j, k) = (double)j;
                    grid_blk.z(i, j, k) = (double)k;
                }
            }
        }
    }

    // 重新计算所有 Block 的 Metrics
    orion::mesh::compute_grid_metrics(grid, fs, params);

    // 验证体积是否为 1.0
    for (int nb : fs.local_block_ids) {
        auto& fs_blk = fs.blocks[nb];
        int ni = fs_blk.prim.dims()[0];
        int nj = fs_blk.prim.dims()[1];
        int nk = fs_blk.prim.dims()[2];
        int ng = fs.ng;

        for (int k = ng; k < nk - ng; ++k) {
            for (int j = ng; j < nj - ng; ++j) {
                for (int i = ng; i < ni - ng; ++i) {
                    double vol = fs_blk.vol(i, j, k);
                    if (std::abs(vol - 1.0) > 1.0e-12) {
                        if (err_count < 5) LOG_FAIL("Unit Cube Vol != 1.0");
                        err_count++;
                    }
                }
            }
        }
    }

    int global_err = 0;
    MPI_Allreduce(&err_count, &global_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_err == 0) {
        LOG_PASS("Perfect Cube Test Passed (All Blocks Modified).");
    } else {
        LOG_FAIL("Perfect Cube Test Failed!");
    }
}

// ---------------------------------------------------------------------------
// 测试 3: 物理边界条件逻辑测试 (全覆盖 + 密度取反检查)
// ---------------------------------------------------------------------------
void test_physical_bc_logic(orion::bc::BCData& bc, 
                            orion::preprocess::FlowFieldSet& fs,
                            orion::core::Params params) // 按值传递
{
    LOG_INFO(">>> TEST 3: Physical Boundary Condition Logic Check (All Types, All Layers)");

    const double R_gas = 287.0; 
    const double gamma = 1.4;
    int err_count = 0;

    // 设置参考来流参数
    params.inflow.roo = 9.99; 
    params.inflow.uoo = 8.88;
    params.inflow.voo = 7.77;
    params.inflow.woo = 6.66;
    params.inflow.poo = 1000.0;
    params.inflow.gama = 1.4;

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        auto& bcb = bc.block_bc[nb];

        int ni = bf.prim.dims()[0];
        int nj = bf.prim.dims()[1];
        int nk = bf.prim.dims()[2];
        int ng = fs.ng;

        // 1. 构造带有梯度的内部流场
        std::fill(bf.prim.hostPtr(), bf.prim.hostPtr() + bf.prim.size(), -999.0);

        for(int k=ng; k<nk-ng; ++k) {
            for(int j=ng; j<nj-ng; ++j) {
                for(int i=ng; i<ni-ng; ++i) {
                    double val_base = 1.0 + (i+j+k)*0.01; 
                    bf.prim(i,j,k,0) = val_base;             
                    bf.prim(i,j,k,1) = 100.0 + i*1.0; 
                    bf.prim(i,j,k,2) = 50.0  + j*1.0;        
                    bf.prim(i,j,k,3) = 20.0  + k*1.0;        
                    bf.prim(i,j,k,4) = 100000.0 + val_base*10; 
                    if (bf.prim.dims()[3] > 5) bf.prim(i,j,k,5) = 300.0 + i*0.5;
                }
            }
        }

        // 2. 遍历边界进行测试
        for (auto& reg : bcb.regions) {
            if (reg.nbt > 0 || reg.bctype == 0) continue;

            // 几何定位
            int dir = reg.s_nd - 1; 
            int ic = (std::abs(reg.s_st[0]) + std::abs(reg.s_ed[0])) / 2;
            int jc = (std::abs(reg.s_st[1]) + std::abs(reg.s_ed[1])) / 2;
            int kc = (std::abs(reg.s_st[2]) + std::abs(reg.s_ed[2])) / 2;
            
            int i_c = ic-1+ng, j_c = jc-1+ng, k_c = kc-1+ng;

            int off = 0;
            if (dir == 0) off = (i_c < ni/2) ? -1 : 1;
            else if (dir == 1) off = (j_c < nj/2) ? -1 : 1;
            else off = (k_c < nk/2) ? -1 : 1;

            int original_bctype = reg.bctype;

            // --- 验证器 Lambda ---
            auto verify_bc = [&](const std::string& case_name, std::function<void(int ig, int jg, int kg, int ii, int ji, int ki, int layer)> checker) {
                for (int g = 1; g <= ng; ++g) {
                    
                    int i_base = i_c, j_base = j_c, k_base = k_c;
                    if (dir==0 && off==-1) i_base = ng;
                    if (dir==0 && off== 1) i_base = ni-ng-1;
                    if (dir==1 && off==-1) j_base = ng;
                    if (dir==1 && off== 1) j_base = nj-ng-1;
                    if (dir==2 && off==-1) k_base = ng;
                    if (dir==2 && off== 1) k_base = nk-ng-1;

                    int shift_g = (off == -1) ? -g : g;  
                    int shift_i = (off == -1) ? g : -g;  

                    int ig = i_base, jg = j_base, kg = k_base;
                    int ii = i_base, ji = j_base, ki = k_base;
                    
                    if (dir==0) { ig += shift_g; ii += shift_i; }
                    if (dir==1) { jg += shift_g; ji += shift_i; }
                    if (dir==2) { kg += shift_g; ki += shift_i; }

                    checker(ig, jg, kg, ii, ji, ki, g);
                }
            };

            // =========================================================
            // [Case 2] Inviscid Wall (Slip)
            // =========================================================
            reg.bctype = 2; params.flowtype.nvis = 0;
            
            verify_bc("Clear", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                bf.prim(ig, jg, kg, 1) = -9999.0; 
                bf.prim(ig, jg, kg, 0) = -9999.0;
            });

            orion::bc::apply_physical_bc(bc, fs, params);
            
            verify_bc("SlipWall", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                double u_g = bf.prim(ig, jg, kg, 1); double u_i = bf.prim(ii, ji, ki, 1);
                double rho_g = bf.prim(ig, jg, kg, 0); double rho_i = bf.prim(ii, ji, ki, 0);
                
                // 1. 速度检查
                bool fail = false;
                if (dir == 0) { // Normal X
                    if (std::abs(u_g + u_i) > 1e-5) fail = true; 
                } else if (dir == 1) { // Normal Y
                    if (std::abs(u_g - u_i) > 1e-5) fail = true;
                }
                if (fail) {
                    LOG_FAIL("[SlipWall] Layer " + std::to_string(g) + " Mismatch!");
                    err_count++;
                }

                // 2. 密度取反检查 (Layer 3)
                bool rho_fail = false;
                if (g == 3) {
                    if (std::abs(rho_g + rho_i) > 1e-5) { // Expect Negative
                        LOG_FAIL("[SlipWall] Layer 3 Density NOT Inverted!");
                        rho_fail = true;
                    }
                } else {
                    if (std::abs(rho_g - rho_i) > 1e-5) { // Expect Positive
                        LOG_FAIL("[SlipWall] Layer " + std::to_string(g) + " Density Mismatch!");
                        rho_fail = true;
                    }
                }
                if (rho_fail) err_count++;
            });

            // =========================================================
            // [Case 2] Viscous Wall (No-Slip)
            // =========================================================
            reg.bctype = 2; params.flowtype.nvis = 1;
            orion::bc::apply_physical_bc(bc, fs, params);
            
            verify_bc("ViscWall", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                double u_g = bf.prim(ig, jg, kg, 1);
                double u_i = bf.prim(ii, ji, ki, 1);
                double rho_g = bf.prim(ig, jg, kg, 0);
                double rho_i = bf.prim(ii, ji, ki, 0); // 绝热壁面 Rho_g = Rho_i (或者计算值)

                if (std::abs(u_g + u_i) > 1e-5) {
                    LOG_FAIL("[ViscWall] Vel Mismatch.");
                    err_count++;
                }

                // [新增] 密度取反检查 (Fortran逻辑：Layer 3 密度取反)
                // 注意：Viscous Wall 计算出的密度可能经过了插值，不一定严格等于 rho_i
                // 但正负号必须对。
                bool rho_fail = false;
                if (g == 3) {
                    if (rho_g > 0.0) { // Should be negative
                        LOG_FAIL("[ViscWall] Layer 3 Density NOT Inverted (Got > 0)!");
                        rho_fail = true;
                    }
                } else {
                    if (rho_g < 0.0) { // Should be positive
                        LOG_FAIL("[ViscWall] Layer " + std::to_string(g) + " Density Negative!");
                        rho_fail = true;
                    }
                }
                if (rho_fail) err_count++;
            });

            // =========================================================
            // [Case 4] Farfield (Supersonic Outflow)
            // =========================================================
            reg.bctype = 4;
            // 构造超声速流出，使其退化为外推
            // 简单起见，不改变内部场，只检查密度是否在 Layer 3 取反
            // (Farfield 如果是亚声速边界，密度会设为 Params 值或内部值)
            orion::bc::apply_physical_bc(bc, fs, params);
            
            verify_bc("Farfield", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                 double rho_g = bf.prim(ig, jg, kg, 0);
                 
                 // [新增] 密度取反检查
                 bool rho_fail = false;
                 if (g == 3) {
                     if (rho_g > 0.0) { 
                         LOG_FAIL("[Farfield] Layer 3 Density NOT Inverted!");
                         rho_fail = true;
                     }
                 } else {
                     if (rho_g < 0.0) {
                         LOG_FAIL("[Farfield] Layer " + std::to_string(g) + " Density Negative!");
                         rho_fail = true;
                     }
                 }
                 if (rho_fail) err_count++;
             });

            // =========================================================
            // [Case 3] Symmetry
            // =========================================================
            reg.bctype = 3;
            orion::bc::apply_physical_bc(bc, fs, params);
            verify_bc("Symmetry", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                double p_g = bf.prim(ig, jg, kg, 4);
                double p_i = bf.prim(ii, ji, ki, 4);
                double rho_g = bf.prim(ig, jg, kg, 0);
                if (std::abs(p_g - p_i) > 1e-5) {
                    LOG_FAIL("[Symmetry] Pressure Mismatch");
                    err_count++;
                }
                // Symmetry: 所有层密度均为正
                if (rho_g < 0.0) {
                    LOG_FAIL("[Symmetry] Density Negative!");
                    err_count++;
                }
            });

            // =========================================================
            // [Case 6] Extrapolation
            // =========================================================
            reg.bctype = 6;
            orion::bc::apply_physical_bc(bc, fs, params);
            verify_bc("Extrap", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                double rho_g = bf.prim(ig, jg, kg, 0);
                // Extrap: 所有层密度均为正
                if (rho_g < 0.0) {
                    LOG_FAIL("[Extrap] Density Negative!");
                    err_count++;
                }
            });

            // =========================================================
            // [Case 5] Freestream
            // =========================================================
            reg.bctype = 5;
            orion::bc::apply_physical_bc(bc, fs, params);
            verify_bc("Freestream", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                double rho_g = bf.prim(ig, jg, kg, 0);
                if (std::abs(rho_g - params.inflow.roo) > 1e-5) {
                    LOG_FAIL("[Freestream] Value Incorrect");
                    err_count++;
                }
                // Freestream: 密度始终为正
                if (rho_g < 0.0) {
                    LOG_FAIL("[Freestream] Density Negative!");
                    err_count++;
                }
            });

            reg.bctype = original_bctype;
            break; 
        }
    }

    int global_err = 0;
    MPI_Allreduce(&err_count, &global_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_err == 0) LOG_PASS("Physical BC Logic (6 Cases, All Layers) Passed.");
    else LOG_FAIL("Physical BC Logic Found Bugs!");
}

// ---------------------------------------------------------------------------
// 测试 4: 来流保持测试 (Flux Consistency)
// 注意：默认关闭，需用户手动开启
// ---------------------------------------------------------------------------
void test_freestream(orion::solver::HaloExchanger& exchanger,
                     orion::bc::BCData& bc, 
                     orion::preprocess::FlowFieldSet& fs,
                     const orion::core::Params& params)
{
    LOG_INFO(">>> TEST 4: Free-stream Preservation (Flux Check)");
    double gamma = 1.4;

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        int n_cells = bf.prim.dims()[0] * bf.prim.dims()[1] * bf.prim.dims()[2];
        
        double rho = 1.0, u = 100.0, v = 20.0, w = 5.0, press = 100000.0;
        
        for(int i=0; i<n_cells; ++i) {
            bf.prim.hostPtr()[i + 0*n_cells] = rho;
            bf.prim.hostPtr()[i + 1*n_cells] = u;
            bf.prim.hostPtr()[i + 2*n_cells] = v;
            bf.prim.hostPtr()[i + 3*n_cells] = w;
            bf.prim.hostPtr()[i + 4*n_cells] = press;
        }
        std::fill(bf.dq.hostPtr(), bf.dq.hostPtr() + bf.dq.size(), 0.0);
    }

    orion::solver::StateUpdater::update_flow_states(fs, params);
    exchanger.exchange_bc(bc, fs);
    orion::solver::StateUpdater::update_flow_states(fs, params);

    orion::solver::InviscidFluxComputer::compute_inviscid_rhs(fs, params);

    double local_max = 0.0;
    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        int ni = bf.dq.dims()[0];
        int nj = bf.dq.dims()[1];
        int nk = bf.dq.dims()[2];
        for (int m=0; m<5; ++m) {
            for(int k=fs.ng; k<nk-fs.ng; ++k)
                for(int j=fs.ng; j<nj-fs.ng; ++j)
                    for(int i=fs.ng; i<ni-fs.ng; ++i)
                        local_max = std::max(local_max, std::abs(bf.dq(i,j,k,m)));
        }
    }

    double global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (global_max < 1.0e-9) {
        LOG_PASS("Free-stream preserved. Flux calculation is consistent.");
    } else {
        if(orion::core::Runtime::is_root()) 
            std::cout << "\033[1;31m[FAIL] Max Residual = " << std::scientific << global_max << "\033[0m" << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    orion::core::Runtime::init(argc, argv);
    
    if (orion::core::Runtime::is_root()) {
        std::cout << "================================================\n";
        std::cout << "     ORION SOLVER - SYSTEM BENCHMARK TOOL       \n";
        std::cout << "================================================\n";
    }

    orion::core::Params params;
    params.load_namelist_file("param.dat");

    orion::mesh::MultiBlockGrid grid;
    orion::bc::BCData bc;
    
    if (orion::core::Runtime::is_root()) {
        grid = orion::mesh::Plot3DReader::read(params.filename.gridname, params.inflow.ndim);
        bc = orion::bc::BCReader::read_parallel_bc_root_only(params.filename.bcname, grid, orion::core::Runtime::nproc, {.verbose=false});
    }
    orion::mesh::distrib_grid_fast(grid, bc.block_pid, orion::core::Runtime::myid, orion::core::Runtime::nproc, {.master=0});
    orion::bc::distrib_bc_fast(bc, orion::core::Runtime::myid, orion::core::Runtime::nproc, {.master=0});
    orion::bc::set_bc_index(bc);
    orion::bc::prepare_bc_topology(bc);

    orion::preprocess::FlowFieldSet fs = orion::preprocess::allocate_other_variable(grid, bc, params, 5, 6);
    
    orion::solver::HaloExchanger exchanger;

    MPI_Barrier(MPI_COMM_WORLD);
    test_halo_exchange(exchanger, bc, fs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    test_metrics(grid, fs, params);
    
    MPI_Barrier(MPI_COMM_WORLD);
    test_physical_bc_logic(bc, fs, params);

    // [OPTIONAL] Test 4: Flux Consistency
    // MPI_Barrier(MPI_COMM_WORLD);
    // test_freestream(exchanger, bc, fs, params);

    orion::core::Runtime::finalize();
    return 0;
}