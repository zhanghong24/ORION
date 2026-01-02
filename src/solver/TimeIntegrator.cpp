#include "solver/TimeIntegrator.hpp"
#include <algorithm>
#include <cmath>
#include <mpi.h> // 需要 MPI_Allreduce 定义

namespace orion::solver {

// ============================================================================
// Helper: Reduce Global DT (虽然 Fortran 逻辑中暂时不用，但为了完整性保留实现)
// ============================================================================
double TimeIntegrator::reduce_global_dt(double local_min_dt) {
    double global_dt = local_min_dt;
    MPI_Allreduce(&local_min_dt, &global_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    return global_dt;
}

// ============================================================================
// Subroutine: calculate_spectrum
// ============================================================================
void TimeIntegrator::calculate_spectrum(orion::preprocess::BlockField& bf, 
                                        const orion::core::Params& params, 
                                        int ng) {
    int ni = bf.prim.dims()[0];
    int nj = bf.prim.dims()[1];
    int nk = bf.prim.dims()[2];

    double reynolds = params.inflow.reynolds;
    double csrv = params.technic.csrv;
    int nvis = params.flowtype.nvis;

    // --- 1. Inviscid Spectral Radius ---
    for (int dir = 0; dir < 3; ++dir) {
        // 遍历所有网格 (包含 Ghost Cells，用于后续插值或通量计算)
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    
                    double u = bf.prim(i, j, k, 1);
                    double v = bf.prim(i, j, k, 2);
                    double w = bf.prim(i, j, k, 3);
                    double c = bf.c(i, j, k);

                    // 获取度量系数
                    // FlowField.hpp 定义: 
                    // 0-2: xi_x, xi_y, xi_z
                    // 3-5: eta_x, eta_y, eta_z
                    // 6-8: zeta_x, zeta_y, zeta_z
                    int m_idx = dir * 3;
                    double mx = bf.metrics(i, j, k, m_idx + 0);
                    double my = bf.metrics(i, j, k, m_idx + 1);
                    double mz = bf.metrics(i, j, k, m_idx + 2);

                    // 逆变速度 U_contra = |u*kx + v*ky + w*kz| + c * |grad k|
                    double U_contra = std::abs(u * mx + v * my + w * mz);
                    double grad_len = std::sqrt(mx * mx + my * my + mz * mz);

                    bf.spec_radius(i, j, k, dir) = U_contra + c * grad_len;
                }
            }
        }
    }

    // --- 2. Viscous Spectral Radius ---
    if (nvis == 1) {
        for (int dir = 0; dir < 3; ++dir) {
            for (int k = 0; k < nk; ++k) {
                for (int j = 0; j < nj; ++j) {
                    for (int i = 0; i < ni; ++i) {
                        
                        int m_idx = dir * 3;
                        double mx = bf.metrics(i, j, k, m_idx + 0);
                        double my = bf.metrics(i, j, k, m_idx + 1);
                        double mz = bf.metrics(i, j, k, m_idx + 2);
                        double grad_sq = mx*mx + my*my + mz*mz;

                        double rho = bf.prim(i, j, k, 0);
                        double vol = bf.vol(i, j, k);
                        double mu = bf.mu(i, j, k); 

                        // Formula: 2 * mu / (Re * rho * Vol) * |grad|^2 * csrv
                        double denom = reynolds * rho * vol + 1.0e-30;
                        double val = (2.0 * mu / denom) * grad_sq * csrv;

                        bf.spec_radius_visc(i, j, k, dir) = val;
                    }
                }
            }
        }
    } else {
        // 如果无粘，清零粘性谱半径
        bf.spec_radius_visc.fill(0.0);
    }
}

// ============================================================================
// Subroutine: localdt0
// ============================================================================
void TimeIntegrator::calculate_local_dt_cell(orion::preprocess::BlockField& bf, 
                                             const orion::core::Params& params, 
                                             int ng) {
    int ni = bf.prim.dims()[0];
    int nj = bf.prim.dims()[1];
    int nk = bf.prim.dims()[2];
    
    double cfl = params.step.cfl;

    // 计算局部时间步长，并存储为 (dt / vol) 格式
    for (int k = ng; k < nk-ng; ++k) {
        for (int j = ng; j < nj-ng; ++j) {
            for (int i = ng; i < ni-ng; ++i) {
                double sum_spec = 0.0;
                for (int dir = 0; dir < 3; ++dir) {
                    sum_spec += bf.spec_radius(i, j, k, dir) + bf.spec_radius_visc(i, j, k, dir);
                }
                
                // 物理时间步长 dt_phys
                double dt_phys = cfl / (sum_spec + 1.0e-30);

                // 存储到 bf.dt 中。注意：为了配合后续逻辑，这里存 dt/vol
                double vol = bf.vol(i, j, k);
                bf.dt(i, j, k) = dt_phys / vol;
            }
        }
    }
}

// ============================================================================
// Main Entry: timestep_tgh
// ============================================================================
void TimeIntegrator::calculate_time_step(orion::preprocess::FlowFieldSet& fs, 
                                         orion::core::Params& params) {
    int ng = fs.ng;
    int ntmst = params.step.ntmst;
    double sml_vol = 1.0e-30; 

    // 1. 基础计算：计算所有 Block 的谱半径和初始 dt (dt/vol)
    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        calculate_spectrum(bf, params, ng);
        calculate_local_dt_cell(bf, params, ng);
    }

    // 2. 根据 ntmst 模式调整
    if (ntmst == 1) {
        /// TODO
    } 
    else {
        // --- Unsteady Mode (Global Time Stepping) ---
        // 逻辑：严格复刻 Fortran，在本地寻找 min(dt_phys)，但不做 MPI 规约
        
        double dtmin = 1000.0; // Fortran Hardcode value

        for (int nb : fs.local_block_ids) {
            auto& bf = fs.blocks[nb];
            int ni = bf.dt.dims()[0];
            int nj = bf.dt.dims()[1];
            int nk = bf.dt.dims()[2];

            // 循环范围：避开 Ghost Cells (Fortran: 2 to N-1)
            // 对应 C++: ng 到 dim-ng
            for (int k = ng; k < nk - ng; ++k) {
                for (int j = ng; j < nj - ng; ++j) {
                    for (int i = ng; i < ni - ng; ++i) {
                        // 还原物理时间： dt_phys = (dt/vol) * vol
                        double dt_phys = bf.dt(i, j, k) * bf.vol(i, j, k);
                        
                        if (dt_phys < dtmin) {
                            dtmin = dt_phys;
                        }
                    }
                }
            }
        }
        
        // 更新全局参数 (注意：这里直接使用本地计算的最小值，未做 Global Reduce)
        params.step.phydtime = dtmin;
    }
}

} // namespace orion::solver