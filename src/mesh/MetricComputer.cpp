#include "mesh/MetricComputer.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <mpi.h>
#include <algorithm>
#include <iomanip>

namespace orion::mesh {

// Internal constants for schemes
static constexpr int NIJK2ND = 4;

namespace {

// ------------------------------------------------------------------
// 1D Processing Helpers (Fortran logic ported)
// ------------------------------------------------------------------

void value_half_node_1d(int nvar, int ni, 
                        const std::vector<double>& q, 
                        std::vector<double>& q_half)
{
    const double A1 = 9.0, B1 = -1.0;
    const double A2 = 5.0, B2 = 15.0, C2 = -5.0, D2 = 1.0;

    auto Q = [&](int v, int i) { return q[v + nvar * i]; };
    auto QH = [&](int v, int i) -> double& { return q_half[v + nvar * i]; };

    if (ni <= NIJK2ND) {
        std::cout << "WARNING ABOUT ERROR" << std::endl;
    } else {
        for (int i = 2; i < ni - 1; ++i) {
            for (int m = 0; m < nvar; ++m) 
                QH(m, i) = (A1 * (Q(m, i-1) + Q(m, i)) + B1 * (Q(m, i+1) + Q(m, i-2))) / 16.0;
        }
        for (int m = 0; m < nvar; ++m) {
            QH(m, 1) = (A2*Q(m,0) + B2*Q(m,1) + C2*Q(m,2) + D2*Q(m,3)) / 16.0;
            QH(m, ni-1) = (A2*Q(m,ni-1) + B2*Q(m,ni-2) + C2*Q(m,ni-3) + D2*Q(m,ni-4)) / 16.0;
            QH(m, 0) = (35.0*Q(m,0) - 35.0*Q(m,1) + 21.0*Q(m,2) - 5.0*Q(m,3)) / 16.0;
            QH(m, ni) = (35.0*Q(m,ni-1) - 35.0*Q(m,ni-2) + 21.0*Q(m,ni-3) - 5.0*Q(m,ni-4)) / 16.0;
        }
    }
}

void flux_dxyz_1d(int nvar, int ni,
                  const std::vector<double>& f,
                  std::vector<double>& df)
{
    auto F = [&](int v, int i) { return f[v + nvar * i]; };
    auto DF = [&](int v, int i) -> double& { return df[v + nvar * i]; };

    // 对应 Fortran nijk2nd = 4
    if (ni <= NIJK2ND) {
        std::cout << "WARNING ABOUT ERROR" << std::endl;
    } else {
        // [Fixed] Internal 6th order
        // C++ loop i corresponds to physical node i.
        // Formula F(i+1) - F(i) calculates derivative centered at Node i.
        // Range: Fortran i=3..ni-2 (Physical 2..ni-3) -> C++ i=2..ni-3
        for (int i = 2; i <= ni - 3; ++i) {
            for (int m = 0; m < nvar; ++m) {
                double t1 = F(m, i+1) - F(m, i);
                double t2 = F(m, i+2) - F(m, i-1);
                double t3 = F(m, i+3) - F(m, i-2);
                DF(m, i) = (2250.0 * t1 - 125.0 * t2 + 9.0 * t3) / 1920.0;
            }
        }
        
        // [Fixed] Boundary Schemes aligned with Fortran FLUX_DXYZ

        // 1. Left Boundary Node 0 (i=0)
        // Fortran: df(1) = (-23*f(0) + 21*f(1) + 3*f(2) - f(3)) / 24
        // C++: F(0) is ghost half, F(1) is first inner half.
        for (int m = 0; m < nvar; ++m) {
            DF(m, 0) = (-23.0*F(m, 0) + 21.0*F(m, 1) + 3.0*F(m, 2) - F(m, 3)) / 24.0;
        }

        // 2. Left Boundary Node 1 (i=1)
        // Fortran: df(2) = (f(0) - 27*f(1) + 27*f(2) - f(3)) / 24
        for (int m = 0; m < nvar; ++m) {
            DF(m, 1) = (F(m, 0) - 27.0*F(m, 1) + 27.0*F(m, 2) - F(m, 3)) / 24.0;
        }

        // 3. Right Boundary Node ni-2 (i=ni-2)
        // Fortran: df(ni-1) = -(f(ni) - 27*f(ni-1) + 27*f(ni-2) - f(ni-3)) / 24
        // C++: F(ni) is ghost half.
        for (int m = 0; m < nvar; ++m) {
            DF(m, ni-2) = -(F(m, ni) - 27.0*F(m, ni-1) + 27.0*F(m, ni-2) - F(m, ni-3)) / 24.0;
        }

        // 4. Right Boundary Node ni-1 (i=ni-1)
        // Fortran: df(ni) = -(-23*f(ni) + 21*f(ni-1) + 3*f(ni-2) - f(ni-3)) / 24
        for (int m = 0; m < nvar; ++m) {
            DF(m, ni-1) = -(-23.0*F(m, ni) + 21.0*F(m, ni-1) + 3.0*F(m, ni-2) - F(m, ni-3)) / 24.0;
        }
    }
}

} // namespace anonymous

void compute_grid_metrics(const MultiBlockGrid& grid,
                          orion::preprocess::FlowFieldSet& fs,
                          const orion::core::Params& params)
{
    const int ng = fs.ng;
    const int nscheme = params.technic.nscheme;
    const int gcl_enabled = params.gcl_cic.gcl;

    if (nscheme != params.technic.nscheme) {
        std::cerr << "[Metric] Warning: scheme mismatch.\n";
    }

    // Process each local block
    for (int nb_idx : fs.local_block_ids)
    {
        // -------------------------------------------------------------
        // Debug Checks: Diagnose the 0x8 Segfault Cause
        // -------------------------------------------------------------
        if (nb_idx < 0 || nb_idx >= grid.nblocks) {
            std::cerr << "FATAL: Block ID " << nb_idx << " out of range (0.." << grid.nblocks-1 << ")\n";
            std::exit(-1);
        }
        // Check Geometry (If empty, accessing kdim crashes at 0x8)
        if (grid.blocks.size() <= (size_t)nb_idx) {
             std::cerr << "FATAL: grid.blocks not resized correctly on rank " << orion::core::Runtime::rank() << "\n";
             std::exit(-1);
        }
        const auto& gb = grid.blocks[nb_idx];
        if (gb.x.size() == 0) {
            std::cerr << "FATAL: Block " << (nb_idx+1) << " Geometry (x) is EMPTY. GridDistributor failed?\n";
            std::exit(-1);
        }

        // Check Metrics (If empty, accessing dims_ crashes at 0x8)
        if (fs.blocks.size() <= (size_t)nb_idx) {
             std::cerr << "FATAL: fs.blocks not resized correctly.\n";
             std::exit(-1);
        }
        auto& fb = fs.blocks[nb_idx];
        if (fb.metrics.size() == 0) {
            std::cerr << "FATAL: Block " << (nb_idx+1) << " Metrics is EMPTY. Allocation in FlowField.cpp failed?\n";
            std::exit(-1);
        }

        // -------------------------------------------------------------
        // Main Computation
        // -------------------------------------------------------------
        const int ni = gb.idim;
        const int nj = gb.jdim;
        const int nk = gb.kdim;

        if (ni <= 2 * ng || nj <= 2 * ng || nk <= 2 * ng) {
            std::cerr << "Error: Grid too small in block " << (nb_idx+1) << "\n";
            continue;
        }

        // Fix Integer Overflow: Use size_t for allocation size
        std::size_t dxyz_len = (std::size_t)18 * ni * nj * nk;
        std::vector<double> dxyz(dxyz_len, 0.0);
        
        // Lambda to access dxyz safely
        auto DXYZ = [&](int comp, int i, int j, int k) -> double& {
            return dxyz[comp + 18 * ((std::size_t)i + ni * ((std::size_t)j + nj * (std::size_t)k))];
        };

        // Buffers for 1D operations
        int nmax = std::max({ni, nj, nk});
        std::vector<double> line_in_3(3 * nmax);
        std::vector<double> line_half_3(3 * (nmax + 1));
        std::vector<double> line_out_3(3 * nmax);
        
        // Pass 1: Simple derivatives
        // 1.1 Xi
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    line_in_3[0 + 3*i] = gb.x(i, j, k);
                    line_in_3[1 + 3*i] = gb.y(i, j, k);
                    line_in_3[2 + 3*i] = gb.z(i, j, k);
                }
                value_half_node_1d(3, ni, line_in_3, line_half_3);
                flux_dxyz_1d(3, ni, line_half_3, line_out_3);
                for (int i = 0; i < ni; ++i) {
                    DXYZ(0, i, j, k) = line_out_3[0 + 3*i]; 
                    DXYZ(3, i, j, k) = line_out_3[1 + 3*i]; 
                    DXYZ(6, i, j, k) = line_out_3[2 + 3*i]; 
                }
            }
        }
        // 1.2 Eta
        for (int k = 0; k < nk; ++k) {
            for (int i = 0; i < ni; ++i) {
                for (int j = 0; j < nj; ++j) {
                    line_in_3[0 + 3*j] = gb.x(i, j, k);
                    line_in_3[1 + 3*j] = gb.y(i, j, k);
                    line_in_3[2 + 3*j] = gb.z(i, j, k);
                }
                value_half_node_1d(3, nj, line_in_3, line_half_3);
                flux_dxyz_1d(3, nj, line_half_3, line_out_3);
                for (int j = 0; j < nj; ++j) {
                    DXYZ(1, i, j, k) = line_out_3[0 + 3*j]; 
                    DXYZ(4, i, j, k) = line_out_3[1 + 3*j]; 
                    DXYZ(7, i, j, k) = line_out_3[2 + 3*j]; 
                }
            }
        }
        // 1.3 Zeta
        for (int j = 0; j < nj; ++j) {
            for (int i = 0; i < ni; ++i) {
                for (int k = 0; k < nk; ++k) {
                    line_in_3[0 + 3*k] = gb.x(i, j, k);
                    line_in_3[1 + 3*k] = gb.y(i, j, k);
                    line_in_3[2 + 3*k] = gb.z(i, j, k);
                }
                value_half_node_1d(3, nk, line_in_3, line_half_3);
                flux_dxyz_1d(3, nk, line_half_3, line_out_3);
                for (int k = 0; k < nk; ++k) {
                    DXYZ(2, i, j, k) = line_out_3[0 + 3*k]; 
                    DXYZ(5, i, j, k) = line_out_3[1 + 3*k]; 
                    DXYZ(8, i, j, k) = line_out_3[2 + 3*k]; 
                }
            }
        }

        // Compute Metrics (Standard)
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    double x_xi  = DXYZ(0,i,j,k); double x_et  = DXYZ(1,i,j,k); double x_zt  = DXYZ(2,i,j,k);
                    double y_xi  = DXYZ(3,i,j,k); double y_et  = DXYZ(4,i,j,k); double y_zt  = DXYZ(5,i,j,k);
                    double z_xi  = DXYZ(6,i,j,k); double z_et  = DXYZ(7,i,j,k); double z_zt  = DXYZ(8,i,j,k);

                    double kcx = y_et * z_zt - z_et * y_zt;
                    double kcy = z_et * x_zt - x_et * z_zt;
                    double kcz = x_et * y_zt - y_et * x_zt;
                    double etx = y_zt * z_xi - z_zt * y_xi;
                    double ety = z_zt * x_xi - x_zt * z_xi;
                    double etz = x_zt * y_xi - y_zt * x_xi;
                    double ctx = y_xi * z_et - z_xi * y_et;
                    double cty = z_xi * x_et - x_xi * z_et;
                    double ctz = x_xi * y_et - y_xi * x_et;
                    double vol = x_xi * kcx + x_et * etx + x_zt * ctx;

                    fb.metrics(i+ng, j+ng, k+ng, 0) = kcx; fb.metrics(i+ng, j+ng, k+ng, 1) = kcy; fb.metrics(i+ng, j+ng, k+ng, 2) = kcz;
                    fb.metrics(i+ng, j+ng, k+ng, 3) = etx; fb.metrics(i+ng, j+ng, k+ng, 4) = ety; fb.metrics(i+ng, j+ng, k+ng, 5) = etz;
                    fb.metrics(i+ng, j+ng, k+ng, 6) = ctx; fb.metrics(i+ng, j+ng, k+ng, 7) = cty; fb.metrics(i+ng, j+ng, k+ng, 8) = ctz;
                    fb.vol(i+ng, j+ng, k+ng) = vol;
                }
            }
        }

        // =========================================================
        // Visbal GCL Correction
        // =========================================================
        if (gcl_enabled == 1) {
            std::vector<double> line_in_6(6 * nmax);
            std::vector<double> line_half_6(6 * (nmax + 1));
            std::vector<double> line_out_6(6 * nmax);

            // Buffer to save original x derivatives for volume calc
            std::vector<double> xkcetct(3 * ni * nj * nk);
            auto XKC = [&](int comp, int i, int j, int k) -> double& {
                return xkcetct[comp + 3 * ((std::size_t)i + ni * ((std::size_t)j + nj * (std::size_t)k))];
            };

            // Step 1: Pre-calculate terms and save original derivatives
            for (int k = 0; k < nk; ++k) {
                for (int j = 0; j < nj; ++j) {
                    for (int i = 0; i < ni; ++i) {
                        double xm = gb.x(i, j, k); double ym = gb.y(i, j, k); double zm = gb.z(i, j, k);
                        
                        // Save x_xi, x_et, x_zt
                        XKC(0,i,j,k) = DXYZ(0,i,j,k); XKC(1,i,j,k) = DXYZ(1,i,j,k); XKC(2,i,j,k) = DXYZ(2,i,j,k);

                        fb.metrics(i+ng, j+ng, k+ng, 0) = DXYZ(0,i,j,k) * ym; // x_xi * ym
                        fb.metrics(i+ng, j+ng, k+ng, 1) = DXYZ(1,i,j,k) * ym; 
                        fb.metrics(i+ng, j+ng, k+ng, 2) = DXYZ(2,i,j,k) * ym; 

                        fb.metrics(i+ng, j+ng, k+ng, 3) = DXYZ(3,i,j,k) * zm; // y_xi * zm
                        fb.metrics(i+ng, j+ng, k+ng, 4) = DXYZ(4,i,j,k) * zm; 
                        fb.metrics(i+ng, j+ng, k+ng, 5) = DXYZ(5,i,j,k) * zm; 

                        fb.metrics(i+ng, j+ng, k+ng, 6) = DXYZ(6,i,j,k) * xm; // z_xi * xm
                        fb.metrics(i+ng, j+ng, k+ng, 7) = DXYZ(7,i,j,k) * xm; 
                        fb.metrics(i+ng, j+ng, k+ng, 8) = DXYZ(8,i,j,k) * xm; 
                    }
                }
            }

            // Step 2: Differentiations
            // Xi
            for (int k = 0; k < nk; ++k) {
                for (int j = 0; j < nj; ++j) {
                    for (int i = 0; i < ni; ++i) {
                        // Gather cty, ctz, kcy, kcz, ety, etz (indices 7,8,1,2,4,5)
                        line_in_6[0 + 6*i] = fb.metrics(i+ng, j+ng, k+ng, 7);
                        line_in_6[1 + 6*i] = fb.metrics(i+ng, j+ng, k+ng, 8);
                        line_in_6[2 + 6*i] = fb.metrics(i+ng, j+ng, k+ng, 1);
                        line_in_6[3 + 6*i] = fb.metrics(i+ng, j+ng, k+ng, 2);
                        line_in_6[4 + 6*i] = fb.metrics(i+ng, j+ng, k+ng, 4);
                        line_in_6[5 + 6*i] = fb.metrics(i+ng, j+ng, k+ng, 5);
                    }
                    value_half_node_1d(6, ni, line_in_6, line_half_6);
                    flux_dxyz_1d(6, ni, line_half_6, line_out_6);
                    for (int i = 0; i < ni; ++i) for(int m=0; m<6; ++m) DXYZ(m, i, j, k) = line_out_6[m + 6*i];
                }
            }
            // Eta
            for (int k = 0; k < nk; ++k) {
                for (int i = 0; i < ni; ++i) {
                    for (int j = 0; j < nj; ++j) {
                        // Gather ctx, ctz, kcx, kcz, etx, etz (indices 6,8,0,2,3,5)
                        line_in_6[0 + 6*j] = fb.metrics(i+ng, j+ng, k+ng, 6);
                        line_in_6[1 + 6*j] = fb.metrics(i+ng, j+ng, k+ng, 8);
                        line_in_6[2 + 6*j] = fb.metrics(i+ng, j+ng, k+ng, 0);
                        line_in_6[3 + 6*j] = fb.metrics(i+ng, j+ng, k+ng, 2);
                        line_in_6[4 + 6*j] = fb.metrics(i+ng, j+ng, k+ng, 3);
                        line_in_6[5 + 6*j] = fb.metrics(i+ng, j+ng, k+ng, 5);
                    }
                    value_half_node_1d(6, nj, line_in_6, line_half_6);
                    flux_dxyz_1d(6, nj, line_half_6, line_out_6);
                    for (int j = 0; j < nj; ++j) for(int m=0; m<6; ++m) DXYZ(m+6, i, j, k) = line_out_6[m + 6*j];
                }
            }
            // Zeta
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    for (int k = 0; k < nk; ++k) {
                        // Gather ctx, cty, kcx, kcy, etx, ety (indices 6,7,0,1,3,4)
                        line_in_6[0 + 6*k] = fb.metrics(i+ng, j+ng, k+ng, 6);
                        line_in_6[1 + 6*k] = fb.metrics(i+ng, j+ng, k+ng, 7);
                        line_in_6[2 + 6*k] = fb.metrics(i+ng, j+ng, k+ng, 0);
                        line_in_6[3 + 6*k] = fb.metrics(i+ng, j+ng, k+ng, 1);
                        line_in_6[4 + 6*k] = fb.metrics(i+ng, j+ng, k+ng, 3);
                        line_in_6[5 + 6*k] = fb.metrics(i+ng, j+ng, k+ng, 4);
                    }
                    value_half_node_1d(6, nk, line_in_6, line_half_6);
                    flux_dxyz_1d(6, nk, line_half_6, line_out_6);
                    for (int k = 0; k < nk; ++k) for(int m=0; m<6; ++m) DXYZ(m+12, i, j, k) = line_out_6[m + 6*k];
                }
            }

            // Step 3: Final Reconstruct
            for (int k = 0; k < nk; ++k) {
                for (int j = 0; j < nj; ++j) {
                    for (int i = 0; i < ni; ++i) {
                        double kcx = DXYZ(17,i,j,k) - DXYZ(11,i,j,k);
                        double kcy = DXYZ(13,i,j,k) - DXYZ(7,i,j,k);
                        double kcz = DXYZ(15,i,j,k) - DXYZ(9,i,j,k);
                        double etx = DXYZ(5,i,j,k) - DXYZ(16,i,j,k);
                        double ety = DXYZ(1,i,j,k) - DXYZ(12,i,j,k);
                        double etz = DXYZ(3,i,j,k) - DXYZ(14,i,j,k);
                        double ctx = DXYZ(10,i,j,k) - DXYZ(4,i,j,k);
                        double cty = DXYZ(6,i,j,k) - DXYZ(0,i,j,k);
                        double ctz = DXYZ(8,i,j,k) - DXYZ(2,i,j,k);

                        fb.metrics(i+ng, j+ng, k+ng, 0) = kcx; fb.metrics(i+ng, j+ng, k+ng, 1) = kcy; fb.metrics(i+ng, j+ng, k+ng, 2) = kcz;
                        fb.metrics(i+ng, j+ng, k+ng, 3) = etx; fb.metrics(i+ng, j+ng, k+ng, 4) = ety; fb.metrics(i+ng, j+ng, k+ng, 5) = etz;
                        fb.metrics(i+ng, j+ng, k+ng, 6) = ctx; fb.metrics(i+ng, j+ng, k+ng, 7) = cty; fb.metrics(i+ng, j+ng, k+ng, 8) = ctz;

                        fb.vol(i+ng, j+ng, k+ng) = XKC(0,i,j,k) * kcx + XKC(1,i,j,k) * etx + XKC(2,i,j,k) * ctx;
                    }
                }
            }
        } // end gcl
    }
}

void check_grid_metrics(const MultiBlockGrid& grid,
                        orion::preprocess::FlowFieldSet& fs,
                        const orion::bc::BCData& bc)
{
    const int ng = fs.ng;
    
    // Constants from Fortran logic
    const double POLE_VOL = 1.0;
    const double SML_VOL = 1.0e-15;  // Adjust epsilon as needed
    const double LARGE_VAL = 1.0e20;

    // Local statistics
    long long negvol_local = 0;
    int is2d_local = 0;
    double minvol_local = LARGE_VAL;
    double maxvol_local = -LARGE_VAL;

    // --- [新增] 用于定位最小值的坐标记录器 ---
    struct MinVolInfo {
        double val = 1.0e20;
        int nb = -1;
        int i = -1;
        int j = -1;
        int k = -1;
    } loc_min;
    // ----------------------------------------

    for (int nb_idx : fs.local_block_ids) {
        auto& fb = fs.blocks[nb_idx];
        const auto& gb = grid.blocks[nb_idx];
        const auto& bcb = bc.block_bc[nb_idx];

        // 1. Override Pole Volumes (BC type 71, 72, 73)
        // Fortran indices are 1-based physical.
        // C++ physical indices start at `ng`.
        // Map: i_cpp = i_fort - 1 + ng
        for (const auto& reg : bcb.regions) {
            if (reg.bctype == 71 || reg.bctype == 72 || reg.bctype == 73) {
                int i_start = reg.s_st[0] - 1 + ng;
                int i_end   = reg.s_ed[0] - 1 + ng;
                int j_start = reg.s_st[1] - 1 + ng;
                int j_end   = reg.s_ed[1] - 1 + ng;
                int k_start = reg.s_st[2] - 1 + ng;
                int k_end   = reg.s_ed[2] - 1 + ng;

                for (int k=k_start; k<=k_end; ++k)
                for (int j=j_start; j<=j_end; ++j)
                for (int i=i_start; i<=i_end; ++i) {
                    fb.vol(i, j, k) = POLE_VOL;
                }
            }
        }

        // 2. Check Loop
        const int ni = gb.idim;
        const int nj = gb.jdim;
        const int nk = gb.kdim;

        if (std::min({ni, nj, nk}) < NIJK2ND) {
            is2d_local = 1;
        }

        int negvol_nb = 0;
        // Iterate over physical domain
        for (int k=0; k<nk; ++k) {
            for (int j=0; j<nj; ++j) {
                for (int i=0; i<ni; ++i) {
                    int ix = i + ng;
                    int iy = j + ng;
                    int iz = k + ng;

                    double val = fb.vol(ix, iy, iz);
                    
                    // --- [新增] 捕捉最小值位置 ---
                    if (val < loc_min.val) {
                        loc_min.val = val;
                        loc_min.nb = nb_idx + 1; // 使用 1-based ID 方便对比
                        loc_min.i = i + 1;       // 使用 1-based Index 方便对比
                        loc_min.j = j + 1;
                        loc_min.k = k + 1;
                    }
                    // ---------------------------

                    if (val < minvol_local) minvol_local = val;
                    if (val > maxvol_local) maxvol_local = val;

                    if (val < SML_VOL) {
                        negvol_nb++;
                        // Warn for first few points
                        if (negvol_nb <= 3) {
                            // Print 1-based indices for consistency with Fortran output
                            std::cout << "Jacobi<0: " << (nb_idx + 1) << " " 
                                      << (i + 1) << " " << (j + 1) << " " << (k + 1) 
                                      << " " << val << "\n";
                        }
                        // Fix volume
                        fb.vol(ix, iy, iz) = SML_VOL;
                    }
                }
            }
        }
        negvol_local += negvol_nb;
    }

    // 3. Global Reduction
    long long negvol_global = 0;
    int is2d_global = 0;
    double minvol_global = 0.0;
    double maxvol_global = 0.0;

    MPI_Reduce(&negvol_local, &negvol_global, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&is2d_local,   &is2d_global,   1, MPI_INT,       MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&minvol_local, &minvol_global, 1, MPI_DOUBLE,    MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&maxvol_local, &maxvol_global, 1, MPI_DOUBLE,    MPI_MAX, 0, MPI_COMM_WORLD);

    // 4. Master Print
    if (orion::core::Runtime::is_root()) {
        if (negvol_global > 0) {
            std::cout << "ERROR(Jacobi<0): " << negvol_global << "\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        } else {
            std::cout << std::scientific << std::setprecision(5);
            std::cout << "$ Interval of Jacobi: (" << minvol_global << "," << maxvol_global << ")\n";
        }

        if (is2d_global > 0) {
            std::cout << "$ The space dimension is 2D!\n";
        }
    }
}

} // namespace orion::mesh