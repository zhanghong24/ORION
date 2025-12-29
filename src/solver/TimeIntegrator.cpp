#include "solver/TimeIntegrator.hpp"
#include "core/Runtime.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

// MPI Support
#ifdef ORION_ENABLE_MPI
#include <mpi.h>
#endif

namespace orion::solver {

// ===========================================================================
// 辅助函数：获取 Metrics (带 Clamping 逻辑)
// ===========================================================================
static void get_metric_clamped(const orion::preprocess::BlockField& bf,
                               int i, int j, int k, int dir, 
                               int i_s, int i_e, int j_s, int j_e, int k_s, int k_e,
                               double& mx, double& my, double& mz, double& mt)
{
    int ic = std::max(i_s, std::min(i_e, i));
    int jc = std::max(j_s, std::min(j_e, j));
    int kc = std::max(k_s, std::min(k_e, k));

    int offset = dir * 3;
    mx = bf.metrics(ic, jc, kc, offset + 0);
    my = bf.metrics(ic, jc, kc, offset + 1);
    mz = bf.metrics(ic, jc, kc, offset + 2);
    mt = 0.0; // Static grid assumption
}

// ===========================================================================
// 1. 无粘谱半径 (Inviscid Spectrum)
// ===========================================================================
static void calculate_inviscid_spectrum(orion::preprocess::BlockField& bf, int ng) {
    const auto& dims = bf.prim.dims();
    int idim = dims[0] - 2 * ng;
    int jdim = dims[1] - 2 * ng;
    int kdim = dims[2] - 2 * ng;

    int i_s = ng, i_e = ng + idim - 1;
    int j_s = ng, j_e = ng + jdim - 1;
    int k_s = ng, k_e = ng + kdim - 1;

    // Loop: Internal + 1 Ghost
    int i_loop_s = i_s - 1, i_loop_e = i_e + 1;
    int j_loop_s = j_s - 1, j_loop_e = j_e + 1;
    int k_loop_s = k_s - 1, k_loop_e = k_e + 1;

    for (int k = k_loop_s; k <= k_loop_e; ++k) {
        for (int j = j_loop_s; j <= j_loop_e; ++j) {
            for (int i = i_loop_s; i <= i_loop_e; ++i) {
                
                double u = bf.prim(i, j, k, 1);
                double v = bf.prim(i, j, k, 2);
                double w = bf.prim(i, j, k, 3);
                double c = bf.c(i, j, k);

                for (int dir = 0; dir < 3; ++dir) {
                    double mx, my, mz, mt;
                    get_metric_clamped(bf, i, j, k, dir, i_s, i_e, j_s, j_e, k_s, k_e, 
                                       mx, my, mz, mt);
                    double U_contra = mx*u + my*v + mz*w + mt;
                    double grad_mag = std::sqrt(mx*mx + my*my + mz*mz);
                    bf.spec_radius(i, j, k, dir) = std::abs(U_contra) + c * grad_mag;
                }
            }
        }
    }
}

// ===========================================================================
// 2. 粘性谱半径 (Viscous Spectrum)
// ===========================================================================
static void calculate_viscous_spectrum(orion::preprocess::BlockField& bf, 
                                       double reynolds, double csrv, int ng) 
{
    const auto& dims = bf.prim.dims();
    int idim = dims[0] - 2 * ng;
    int jdim = dims[1] - 2 * ng;
    int kdim = dims[2] - 2 * ng;

    int i_s = ng, i_e = ng + idim - 1;
    int j_s = ng, j_e = ng + jdim - 1;
    int k_s = ng, k_e = ng + kdim - 1;

    double small = 1.0e-30;

    auto process_point = [&](int i, int j, int k, int i_m, int j_m, int k_m) {
        double rho = bf.prim(i, j, k, 0);   
        double vol = bf.vol(i_m, j_m, k_m); 
        double vis = bf.mu(i_m, j_m, k_m);  
        
        double rm_vol = rho * vol;
        double coef = 2.0 * vis / (reynolds * rm_vol + small);
        double coef1 = csrv * coef;

        for (int dir = 0; dir < 3; ++dir) {
            double mx, my, mz, mt;
            int offset = dir * 3;
            mx = bf.metrics(i_m, j_m, k_m, offset + 0);
            my = bf.metrics(i_m, j_m, k_m, offset + 1);
            mz = bf.metrics(i_m, j_m, k_m, offset + 2);
            double grad_sq = mx*mx + my*my + mz*mz;
            bf.spec_radius_visc(i, j, k, dir) = coef1 * grad_sq;
        }
    };

    // Internal Loop
    for (int k = k_s; k <= k_e; ++k) {
        for (int j = j_s; j <= j_e; ++j) {
            for (int i = i_s; i <= i_e; ++i) {
                process_point(i, j, k, i, j, k);
            }
        }
    }
    // I-Boundary (Ghost)
    int i_bounds[] = {i_s - 1, i_e + 1};
    for (int i : i_bounds) {
        int ii = std::max(i_s, std::min(i_e, i)); 
        for (int k = k_s; k <= k_e; ++k) {
            for (int j = j_s; j <= j_e; ++j) {
                process_point(i, j, k, ii, j, k);
            }
        }
    }
    // J-Boundary (Ghost)
    int j_bounds[] = {j_s - 1, j_e + 1};
    for (int j : j_bounds) {
        int jj = std::max(j_s, std::min(j_e, j));
        for (int k = k_s; k <= k_e; ++k) {
            for (int i = i_s; i <= i_e; ++i) {
                process_point(i, j, k, i, jj, k);
            }
        }
    }
    // K-Boundary (Ghost)
    int k_bounds[] = {k_s - 1, k_e + 1};
    for (int k : k_bounds) {
        int kk = std::max(k_s, std::min(k_e, k));
        for (int j = j_s; j <= j_e; ++j) {
            for (int i = i_s; i <= i_e; ++i) {
                process_point(i, j, k, i, j, kk);
            }
        }
    }
}

// ===========================================================================
// 3. 清零粘性谱半径
// ===========================================================================
static void clear_viscous_spectrum(orion::preprocess::BlockField& bf) {
    bf.spec_radius_visc.fill(0.0);
}

// ===========================================================================
// Spectrum 计算入口
// ===========================================================================
void TimeIntegrator::calculate_spectrum(orion::preprocess::BlockField& bf, 
                                        const orion::core::Params& params,
                                        int ng) 
{
    // [修复] 不再推断 ng，直接使用传入的 ng
    
    // 1. 无粘谱半径
    calculate_inviscid_spectrum(bf, ng);

    // 2. 粘性谱半径
    int nvis = params.flowtype.nvis;
    double csrv = params.technic.csrv; 
    double reynolds = params.inflow.reynolds;

    if (nvis == 1) {
        if (csrv > 0.0) {
            calculate_viscous_spectrum(bf, reynolds, csrv, ng);
        } else {
            clear_viscous_spectrum(bf);
        }
    } else {
        clear_viscous_spectrum(bf);
    }
}

// ===========================================================================
// 4. Local DT Calculation
// ===========================================================================
void TimeIntegrator::calculate_local_dt_cell(orion::preprocess::BlockField& bf, 
                                             const orion::core::Params& params,
                                             int ng) 
{
    const auto& dims = bf.prim.dims();
    // [修复] 不再推断 ng
    
    int idim = dims[0] - 2 * ng;
    int jdim = dims[1] - 2 * ng;
    int kdim = dims[2] - 2 * ng;

    // 参数读取
    double cfl = params.step.cfl; 
    double timedt_rate = params.step.timedt_rate;
    int method = params.control.method;
    double small = 1.0e-30;

    int cmethod = 1 - method;
    
    // Internal loop based on Fortran method logic
    int i_s = ng, i_e = ng + idim - 1 - cmethod;
    int j_s = ng, j_e = ng + jdim - 1 - cmethod;
    int k_s = ng, k_e = ng + kdim - 1 - cmethod;

    for (int k = k_s; k <= k_e; ++k) {
        for (int j = j_s; j <= j_e; ++j) {
            for (int i = i_s; i <= i_e; ++i) {
                
                double ra = bf.spec_radius(i, j, k, 0);
                double rb = bf.spec_radius(i, j, k, 1);
                double rc = bf.spec_radius(i, j, k, 2);

                double rva = bf.spec_radius_visc(i, j, k, 0);
                double rvb = bf.spec_radius_visc(i, j, k, 1);
                double rvc = bf.spec_radius_visc(i, j, k, 2);
                double rv = rva + rvb + rvc;

                double rabc = ra + rb + rc + rv;
                double dt_cfl = cfl / (rabc + small);

                if (timedt_rate > small) {
                    double vol = bf.vol(i, j, k);
                    double dt_limit = timedt_rate / vol;
                    bf.dt(i, j, k) = std::min(dt_cfl, dt_limit);
                } else {
                    bf.dt(i, j, k) = dt_cfl;
                }
            }
        }
    }
}

// ===========================================================================
// Main Function
// ===========================================================================
void TimeIntegrator::calculate_time_step(orion::preprocess::FlowFieldSet& fs, 
                                         orion::core::Params& params)
{
    // [新增] 获取全局统一的 ng
    int ng = fs.ng;

    // 1. 计算每个 Block 的谱半径和初始 DT
    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        calculate_spectrum(bf, params, ng);
        calculate_local_dt_cell(bf, params, ng);
    }

    // 2. 模式选择
    int ntmst = params.step.ntmst;
    double dt_min_limit = 1.0e12; 
    double small_vol = 1.0e-20;   

    if (ntmst == 1) {
        // --- Local Time Stepping (Steady State) ---
        for (int nb : fs.local_block_ids) {
            auto& bf = fs.blocks[nb];
            const auto& dims = bf.dt.dims();
            int nx = dims[0], ny = dims[1], nz = dims[2];
            
            // Loop internal only? Fortran seems to loop ni/nj/nk which usually is internal.
            // Using ng to define internal range.
            
            for (int k = ng; k < nz - ng; ++k) {
                for (int j = ng; j < ny - ng; ++j) {
                    for (int i = ng; i < nx - ng; ++i) {
                        double vol = bf.vol(i, j, k);
                        bf.dt(i, j, k) = dt_min_limit / std::max(vol, small_vol);
                    }
                }
            }
        }
    } else {
        // --- Global Time Stepping (Unsteady) ---
        // [用户要求]：Global DT 部分暂时留空
    }
}

double TimeIntegrator::reduce_global_dt(double local_min_dt) {
    double global_min = local_min_dt;
#ifdef ORION_ENABLE_MPI
    MPI_Allreduce(&local_min_dt, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
    return global_min;
}

} // namespace orion::solver