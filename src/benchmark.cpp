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
    orion::preprocess::FlowFieldSet fs = orion::preprocess::allocate_other_variable(grid, bc, params, 2, 6);
    
    orion::solver::HaloExchanger exchanger;

    MPI_Barrier(MPI_COMM_WORLD);
    test_halo_exchange(exchanger, bc, fs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    test_metrics(grid, fs, params);
    
    MPI_Barrier(MPI_COMM_WORLD);
    test_physical_bc_logic(bc, fs, params);

    MPI_Barrier(MPI_COMM_WORLD);
    test_state_update(fs, params);

    orion::core::Runtime::finalize();
    return 0;
}