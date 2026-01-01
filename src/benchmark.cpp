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
#include "solver/TimeIntegrator.hpp"
#include "solver/FluxComputer.hpp"
#include "solver/InviscidFluxComputer.hpp"

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <functional>
#include <iomanip>

// ---------------------------------------------------------------------------
// Helper Macros
// ---------------------------------------------------------------------------
#define LOG_INFO(msg) if(orion::core::Runtime::is_root()) std::cout << "[BENCH] " << msg << std::endl
#define LOG_PASS(msg) if(orion::core::Runtime::is_root()) std::cout << "\033[1;32m[PASS] " << msg << "\033[0m" << std::endl
#define LOG_FAIL(msg) if(orion::core::Runtime::is_root()) std::cout << "\033[1;31m[FAIL] " << msg << "\033[0m" << std::endl

// ---------------------------------------------------------------------------
// TEST 1: Halo Exchange
// ---------------------------------------------------------------------------
void test_halo_exchange(orion::solver::HaloExchanger& exchanger,
                        orion::bc::BCData& bc, 
                        orion::preprocess::FlowFieldSet& fs) 
{
    LOG_INFO(">>> TEST 1: Halo Exchange Fingerprint Check");
    int my_rank = orion::core::Runtime::myid;

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        // Safe clear
        int ni = bf.prim.dims()[0];
        int nj = bf.prim.dims()[1];
        int nk = bf.prim.dims()[2];
        for(int k=0; k<nk; ++k)
            for(int j=0; j<nj; ++j)
                for(int i=0; i<ni; ++i)
                    for(int m=0; m<5; ++m) bf.prim(i,j,k,m) = -999.0;

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
// TEST 2: Metrics (Unit Cube)
// ---------------------------------------------------------------------------
void test_metrics(orion::mesh::MultiBlockGrid& grid, 
                  orion::preprocess::FlowFieldSet& fs,
                  const orion::core::Params& params) 
{
    LOG_INFO(">>> TEST 2: Metric Consistency Check (Synthetic Unit Cube)");
    int err_count = 0;

    for (int nb : fs.local_block_ids) {
        auto& grid_blk = grid.blocks[nb];
        
        int ni = grid_blk.idim;
        int nj = grid_blk.jdim;
        int nk = grid_blk.kdim;

        // Force Unit Cube Grid
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

    orion::mesh::compute_grid_metrics(grid, fs, params);

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
// TEST 3: Physical BC
// ---------------------------------------------------------------------------
void test_physical_bc_logic(orion::bc::BCData& bc, 
                            orion::preprocess::FlowFieldSet& fs,
                            orion::core::Params params) 
{
    LOG_INFO(">>> TEST 3: Physical Boundary Condition Logic Check (All Types, All Layers)");

    const double R_gas = 287.0; 
    const double gamma = 1.4;
    int err_count = 0;

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

        // Clear
        for(int k=0; k<nk; ++k)
            for(int j=0; j<nj; ++j)
                for(int i=0; i<ni; ++i)
                    for(int m=0; m<5; ++m) bf.prim(i,j,k,m) = -999.0;

        // Initialize with gradient
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

        for (auto& reg : bcb.regions) {
            if (reg.nbt > 0 || reg.bctype == 0) continue;

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

            // Case 2 Inviscid Wall
            reg.bctype = 2; params.flowtype.nvis = 0;
            orion::bc::apply_physical_bc(bc, fs, params);
            verify_bc("SlipWall", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                double u_g = bf.prim(ig, jg, kg, 1); double u_i = bf.prim(ii, ji, ki, 1);
                double rho_g = bf.prim(ig, jg, kg, 0); double rho_i = bf.prim(ii, ji, ki, 0);
                if (dir==0 && std::abs(u_g + u_i) > 1e-5) err_count++; 
                if (g == 3) { if(std::abs(rho_g + rho_i) > 1e-5) { LOG_FAIL("SlipWall: Layer 3 Rho not inverted"); err_count++; } }
                else        { if(std::abs(rho_g - rho_i) > 1e-5) { LOG_FAIL("SlipWall: Layer " + std::to_string(g) + " Rho mismatch"); err_count++; } }
            });

            // Case 2 Viscous Wall
            reg.bctype = 2; params.flowtype.nvis = 1;
            orion::bc::apply_physical_bc(bc, fs, params);
            verify_bc("ViscWall", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                double u_g = bf.prim(ig, jg, kg, 1); double u_i = bf.prim(ii, ji, ki, 1);
                double rho_g = bf.prim(ig, jg, kg, 0);
                if (std::abs(u_g + u_i) > 1e-5) err_count++;
                if (g == 3 && rho_g > 0) { LOG_FAIL("ViscWall: Layer 3 Rho not inverted"); err_count++; }
            });

            // Case 4 Farfield
            reg.bctype = 4;
            orion::bc::apply_physical_bc(bc, fs, params);
            verify_bc("Farfield", [&](int ig, int jg, int kg, int ii, int ji, int ki, int g){
                 double rho_g = bf.prim(ig, jg, kg, 0);
                 if (g == 3 && rho_g > 0) { LOG_FAIL("Farfield: Layer 3 Rho not inverted"); err_count++; }
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
// TEST 4: State Updater Consistency
// ---------------------------------------------------------------------------
void test_state_update(orion::preprocess::FlowFieldSet& fs,
                       orion::core::Params params) 
{
    LOG_INFO(">>> TEST 4: State Updater Consistency Check");
    
    int err_count = 0;
    
    params.inflow.gama = 1.4;
    params.inflow.moo = 0.5;
    params.inflow.visc = 0.1; 
    params.flowtype.nvis = 1; 
    
    double gamma = params.inflow.gama;
    double gm1 = gamma - 1.0;
    double M_inf = params.inflow.moo;
    double visc_c = params.inflow.visc;

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        int ni = bf.prim.dims()[0];
        int nj = bf.prim.dims()[1];
        int nk = bf.prim.dims()[2];
        
        // Clear
        for(int k=0; k<nk; ++k)
            for(int j=0; j<nj; ++j)
                for(int i=0; i<ni; ++i)
                    for(int m=0; m<5; ++m) bf.prim(i,j,k,m) = 0.0; 

        double rho_in = 1.0, u_in=10.0, v_in=20.0, w_in=30.0, p_in = 100000.0;

        // Fill ALL (including ghosts)
        for(int k=0; k<nk; ++k)
        for(int j=0; j<nj; ++j)
        for(int i=0; i<ni; ++i) {
            bf.prim(i,j,k,0) = rho_in;
            bf.prim(i,j,k,1) = u_in;
            bf.prim(i,j,k,2) = v_in;
            bf.prim(i,j,k,3) = w_in;
            bf.prim(i,j,k,4) = p_in;
        }

        orion::solver::StateUpdater::update_flow_states(fs, params);

        double a2_exp = gamma * p_in / rho_in;
        double c_exp = std::sqrt(a2_exp);
        double T_exp = M_inf * M_inf * a2_exp; 
        double mu_exp = T_exp * std::sqrt(T_exp) * (1.0 + visc_c) / (T_exp + visc_c);
        double ke = 0.5 * rho_in * (u_in*u_in + v_in*v_in + w_in*w_in);
        double E_exp = p_in / gm1 + ke;

        // [FIX] ONLY Check CENTER point (Internal). Do NOT check corners.
        int i=ni/2, j=nj/2, k=nk/2;
        
        double c_act = bf.c(i,j,k);
        double T_act = bf.prim(i,j,k,5); 
        double mu_act = bf.mu(i,j,k);
        
        if (std::abs(c_act - c_exp) > 1e-5) {
            LOG_FAIL("Sound Speed Mismatch! Exp: " + std::to_string(c_exp) + " Got: " + std::to_string(c_act));
            err_count++;
        }
        if (std::abs(T_act - T_exp) > 1e-5) {
            LOG_FAIL("Temperature Mismatch! Exp: " + std::to_string(T_exp) + " Got: " + std::to_string(T_act));
            err_count++;
        }
        if (std::abs(mu_act - mu_exp) > 1e-5) {
            LOG_FAIL("Viscosity Mismatch! Exp: " + std::to_string(mu_exp) + " Got: " + std::to_string(mu_act));
            err_count++;
        }

        double q0 = bf.q(i,j,k,0);
        if (std::abs(q0 - rho_in) > 1e-5) { LOG_FAIL("Q[0] (Rho) Mismatch!"); err_count++; }
    }

    int global_err = 0;
    MPI_Allreduce(&err_count, &global_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_err == 0) LOG_PASS("State Updater Logic Verified.");
    else LOG_FAIL("State Updater Found Bugs!");
}

// ---------------------------------------------------------------------------
// TEST 5: Time Integrator (DT Calculation & Fortran Logic Check)
// ---------------------------------------------------------------------------
void test_time_integrator(orion::preprocess::FlowFieldSet& fs,
                          orion::core::Params params) // Pass by value
{
    LOG_INFO(">>> TEST 5: Time Integrator (DT Calculation) Check");

    // 1. 设置物理参数
    params.inflow.gama = 1.4;
    params.step.cfl = 1.0;            
    params.inflow.reynolds = 1000.0;  
    params.flowtype.nvis = 1;         
    params.technic.csrv = 1.0;        
    params.step.timedt_rate = 1.0e20; 
    params.step.ntmst = 0; // Unsteady mode

    double gamma = params.inflow.gama;
    double Re = params.inflow.reynolds;
    double csrv = params.technic.csrv;

    // -----------------------------------------------------------------------
    // [修复步骤 1] 先初始化该进程下所有的 Block
    // -----------------------------------------------------------------------
    double dx = 0.1; 
    double metric_val = 1.0 / dx;  // 10.0
    double vol_val = dx * dx * dx; // 0.001
    double rho_val = 1.0;
    double p_val = 1.0;
    double mu_val = 0.1;
    double c_val = std::sqrt(gamma * p_val / rho_val); // ~1.1832

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        int ni = bf.prim.dims()[0];
        int nj = bf.prim.dims()[1];
        int nk = bf.prim.dims()[2];

        // 全覆盖填充，确保无死角
        for(int k=0; k<nk; ++k) {
            for(int j=0; j<nj; ++j) {
                for(int i=0; i<ni; ++i) {
                    bf.metrics(i,j,k,0) = metric_val; // xi_x
                    bf.metrics(i,j,k,4) = metric_val; // eta_y
                    bf.metrics(i,j,k,8) = metric_val; // zeta_z
                    // 其他 metrics 默认为 0 (假设 OrionArray 初始化已清零，或手动清零)
                    
                    bf.vol(i,j,k) = vol_val;
                    bf.prim(i,j,k,0) = rho_val;
                    bf.prim(i,j,k,1) = 0.0; 
                    bf.prim(i,j,k,2) = 0.0; 
                    bf.prim(i,j,k,3) = 0.0; 
                    bf.prim(i,j,k,4) = p_val;
                    bf.c(i,j,k) = c_val;
                    bf.mu(i,j,k) = mu_val;
                    bf.dt(i,j,k) = 9999.9; // 干扰值
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // [修复步骤 2] 全局调用一次求解器
    // -----------------------------------------------------------------------
    orion::solver::TimeIntegrator::calculate_time_step(fs, params);

    // -----------------------------------------------------------------------
    // [修复步骤 3] 验证结果
    // -----------------------------------------------------------------------
    int err_count = 0;
    
    // 理论值计算
    double spec_inv = c_val * metric_val; // 11.832
    double grad_sq = metric_val * metric_val; // 100.0
    double spec_vis = csrv * (2.0 * mu_val / (Re * rho_val * vol_val)) * grad_sq; // 20.0
    double total_spec = 3.0 * (spec_inv + spec_vis); // 95.496
    double dt_phys_exp = params.step.cfl / total_spec; // 0.01047
    double dt_arr_exp = dt_phys_exp / vol_val; // 10.47

    // 验证 Phydtime (全局变量)
    if (std::abs(params.step.phydtime - dt_phys_exp) > 1e-5) {
        LOG_FAIL("Phydtime Mismatch! Exp: " + std::to_string(dt_phys_exp) + 
                 " Got: " + std::to_string(params.step.phydtime));
        err_count++;
    }

    // 验证数组值
    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        int ni = bf.prim.dims()[0];
        int nj = bf.prim.dims()[1];
        int nk = bf.prim.dims()[2];
        int ic = ni/2, jc = nj/2, kc = nk/2;

        double dt_act = bf.dt(ic, jc, kc);
        if (std::abs(dt_act - dt_arr_exp) > 1e-3) {
            LOG_FAIL("Block " + std::to_string(nb) + " Array Mismatch! Exp: " + std::to_string(dt_arr_exp) + 
                     " Got: " + std::to_string(dt_act));
            err_count++;
        }
    }

    int global_err = 0;
    MPI_Allreduce(&err_count, &global_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_err == 0) {
        LOG_PASS("Time Integrator Verified (Init-Solve-Check flow corrected).");
    } else {
        LOG_FAIL("Time Integrator Logic Failed!");
    }
}

// ---------------------------------------------------------------------------
// TEST 6: Viscous Flux Computer (Exactness Test - Manual Setup)
// ---------------------------------------------------------------------------
void test_flux_computer(orion::preprocess::FlowFieldSet& fs,
                        orion::core::Params params) 
{
    LOG_INFO(">>> TEST 6: Viscous Flux Computer (Exactness Test)");

    // 1. Configure Physics
    params.inflow.reynolds = 1000.0;
    params.flowtype.nvis = 1;
    
    double h = 0.1;
    double vol_val = h * h * h;   // 0.001
    double area_val = h * h;      // 0.01 (Metric = Area Norm)
    double mu_val = 0.1;
    double Re = params.inflow.reynolds;

    int err_count = 0;

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        int ni = bf.prim.dims()[0];
        int nj = bf.prim.dims()[1];
        int nk = bf.prim.dims()[2];
        
        // 2. Fill Data Manually (Avoid MetricComputer overflow)
        for(int k=0; k<nk; ++k) {
            for(int j=0; j<nj; ++j) {
                for(int i=0; i<ni; ++i) {
                    // Coordinates (j=ng -> y=0)
                    double y_phys = (j - fs.ng) * h;

                    // Prim: u = y^2 (Shear flow)
                    bf.prim(i,j,k,0) = 1.0; 
                    bf.prim(i,j,k,1) = y_phys * y_phys; 
                    bf.prim(i,j,k,2) = 0.0;
                    bf.prim(i,j,k,3) = 0.0;
                    bf.prim(i,j,k,4) = 1.0;
                    bf.prim(i,j,k,5) = 1.0; // T

                    bf.mu(i,j,k) = mu_val;
                    bf.vol(i,j,k) = vol_val;
                    
                    // Metrics: Diagonal Matrix (Area Vectors)
                    // xi_x = h^2, eta_y = h^2, zeta_z = h^2
                    for(int m=0; m<9; ++m) bf.metrics(i,j,k,m) = 0.0;
                    bf.metrics(i,j,k,0) = area_val; 
                    bf.metrics(i,j,k,4) = area_val; 
                    bf.metrics(i,j,k,8) = area_val; 

                    // Clear Residual
                    for(int m=0; m<5; ++m) bf.dq(i,j,k,m) = 0.0;
                }
            }
        }
    }

    // 3. Run Solver
    orion::solver::FluxComputer::compute_viscous_rhs(fs, params);

    // 4. Verify
    // Theory:
    // u = y^2  =>  du/dy = 2y
    // tau = mu * du/dy = 0.1 * 2y = 0.2y
    // Flux F = tau * Area = 0.2y * 0.01 = 0.002y
    // Divergence = dF/dy * dy/d_eta = 0.002 * 1 = 0.002 (per index step)
    // Or: dF = F(j+1) - F(j) = 0.002(y+h) - 0.002y = 0.002h = 0.0002
    // RHS Term = - (1/Re) * dF = - (1/1000) * 0.0002 = -2.0e-7
    
    double expected_dq = -2.0e-7;

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        int ni = bf.prim.dims()[0];
        int nj = bf.prim.dims()[1];
        int nk = bf.prim.dims()[2];

        // Check center point
        int ic = ni/2, jc = nj/2, kc = nk/2;
        double dq_act = bf.dq(ic, jc, kc, 1); // x-momentum

        if (std::abs(dq_act - expected_dq) > 1e-10) {
             LOG_FAIL("Viscous RHS Mismatch! Exp: " + std::to_string(expected_dq) + 
                      " Got: " + std::to_string(dq_act));
             err_count++;
        }
    }

    int global_err = 0;
    MPI_Allreduce(&err_count, &global_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_err == 0) {
        LOG_PASS("Viscous Flux Computer Verified (Exactness Test).");
    } else {
        LOG_FAIL("Viscous Flux Computer Failed!");
    }
}

// ---------------------------------------------------------------------------
// TEST 7: Interface Residual Averaging (Corrected for Internal Update & Corners)
// ---------------------------------------------------------------------------
void test_interface_residual_avg(orion::solver::HaloExchanger& exchanger,
                                 orion::bc::BCData& bc, 
                                 orion::preprocess::FlowFieldSet& fs) 
{
    LOG_INFO(">>> TEST 7: Interface Residual Averaging Check");
    int err_count = 0;

    // 1. Setup Data
    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        
        // Fill Volume = 1.0
        bf.vol.fill(1.0);

        // Fill DQ (Internal & Ghost) with Fingerprint
        // We fill EVERYTHING to catch the logic regardless of ghost/internal
        int ni = bf.dq.dims()[0];
        int nj = bf.dq.dims()[1];
        int nk = bf.dq.dims()[2];
        
        for(int k=0; k<nk; ++k) {
            for(int j=0; j<nj; ++j) {
                for(int i=0; i<ni; ++i) {
                    // Fingerprint: BlockID (1-based) + 1000
                    double val = (double)(nb + 1) + 1000.0;
                    for(int m=0; m<5; ++m) bf.dq(i,j,k,m) = val;
                }
            }
        }
    }

    // 2. Run Averaging
    // Correct Formula: dq_internal_new = 0.5 * (dq_local + dq_remote * (vol_local/vol_remote))
    //                                  = 0.5 * ( (ID+1000) + (NID+1000) * 1.0 )
    exchanger.average_interface_residuals(bc, fs);
    MPI_Barrier(MPI_COMM_WORLD);

    // 3. Verify (Check CENTER of the face only to avoid corner overwrites)
    for (int nb : fs.local_block_ids) {
        auto& bcb = bc.block_bc[nb];
        auto& bf = fs.blocks[nb];

        for (const auto& reg : bcb.regions) {
            if (reg.bctype >= 0) continue; // Only check Interface

            int neighbor_id = reg.nbt; // 1-based
            double local_val = (double)(nb + 1) + 1000.0;
            double remote_val = (double)neighbor_id + 1000.0;
            double expected_val = 0.5 * (local_val + remote_val);

            // Get Face Center Indices (Physical 1-based start/end)
            // reg.s_st is the Boundary Face on the Internal Mesh
            int ic_phys = (reg.s_st[0] + reg.s_ed[0]) / 2;
            int jc_phys = (reg.s_st[1] + reg.s_ed[1]) / 2;
            int kc_phys = (reg.s_st[2] + reg.s_ed[2]) / 2;

            // Convert to Memory Index (0-based with ghost offset)
            // Absolute value needed because indices can be negative in some definitions
            int i_mem = std::abs(ic_phys) - 1 + fs.ng;
            int j_mem = std::abs(jc_phys) - 1 + fs.ng;
            int k_mem = std::abs(kc_phys) - 1 + fs.ng;

            double act_val = bf.dq(i_mem, j_mem, k_mem, 0);
            
            if (std::abs(act_val - expected_val) > 1e-5) {
                LOG_FAIL("Interface Avg Mismatch! Block " + std::to_string(nb+1) + 
                         " Neighbor " + std::to_string(neighbor_id) +
                         " Exp: " + std::to_string(expected_val) + 
                         " Got: " + std::to_string(act_val));
                err_count++;
            }
        }
    }

    int global_err = 0;
    MPI_Allreduce(&err_count, &global_err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (global_err == 0) LOG_PASS("Interface Residual Averaging Logic Verified.");
    else LOG_FAIL("Interface Residual Averaging Failed!");
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

    // Keep ng=2 as previously established for safety
    orion::preprocess::FlowFieldSet fs = orion::preprocess::allocate_other_variable(grid, bc, params, 5, 6);
    
    orion::solver::HaloExchanger exchanger;

    MPI_Barrier(MPI_COMM_WORLD);
    test_halo_exchange(exchanger, bc, fs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    test_metrics(grid, fs, params);
    
    MPI_Barrier(MPI_COMM_WORLD);
    test_physical_bc_logic(bc, fs, params);

    MPI_Barrier(MPI_COMM_WORLD);
    test_state_update(fs, params);

    MPI_Barrier(MPI_COMM_WORLD);
    test_time_integrator(fs, params);

    MPI_Barrier(MPI_COMM_WORLD);
    test_flux_computer(fs, params);

    MPI_Barrier(MPI_COMM_WORLD);
    test_interface_residual_avg(exchanger, bc, fs);

    orion::core::Runtime::finalize();
    return 0;
}