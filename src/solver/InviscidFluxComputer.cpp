#include "solver/InviscidFluxComputer.hpp"
#include "core/Runtime.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace orion::solver {

// ===========================================================================
// 辅助函数：计算物理通量 F(Q)
// ===========================================================================
static void calc_physical_flux(const std::vector<double>& q, double gamma, 
                               double nx, double ny, double nz, double nt,
                               std::vector<double>& f)
{
    double rho = q[0];
    double u   = q[1];
    double v   = q[2];
    double w   = q[3];
    double p   = q[4];

    double V_contra = u*nx + v*ny + w*nz + nt; 
    double E = p / (gamma - 1.0) + 0.5 * rho * (u*u + v*v + w*w);
    double H = (E + p) / rho;

    f[0] = rho * V_contra;
    f[1] = rho * u * V_contra + p * nx;
    f[2] = rho * v * V_contra + p * ny;
    f[3] = rho * w * V_contra + p * nz;
    f[4] = rho * H * V_contra + p * nt;
}

// ===========================================================================
// 黎曼求解器：Roe
// ===========================================================================
static void flux_Roe(std::vector<double> ql, std::vector<double> qr, 
                     double nx, double ny, double nz, double nt,
                     double efix, double gamma,
                     std::vector<double>& f)
{
    int nl = 5;
    std::vector<double> fl(nl), fr(nl), dq(nl);
    
    calc_physical_flux(ql, gamma, nx, ny, nz, nt, fl);
    calc_physical_flux(qr, gamma, nx, ny, nz, nt, fr);

    for(int m=0; m<nl; ++m) dq[m] = qr[m] - ql[m];

    double gamma1 = gamma - 1.0;
    double gamma2 = gamma / gamma1;
    
    double hl = gamma2 * ql[4]/ql[0] + 0.5*(ql[1]*ql[1] + ql[2]*ql[2] + ql[3]*ql[3]);
    double hr = gamma2 * qr[4]/qr[0] + 0.5*(qr[1]*qr[1] + qr[2]*qr[2] + qr[3]*qr[3]);
    
    double sqrt_rL = std::sqrt(ql[0]);
    double sqrt_rR = std::sqrt(qr[0]);
    double inv_den = 1.0 / (sqrt_rL + sqrt_rR);

    double rho_roe = sqrt_rL * sqrt_rR;
    double u_roe = (sqrt_rL * ql[1] + sqrt_rR * qr[1]) * inv_den;
    double v_roe = (sqrt_rL * ql[2] + sqrt_rR * qr[2]) * inv_den;
    double w_roe = (sqrt_rL * ql[3] + sqrt_rR * qr[3]) * inv_den;
    double h_roe = (sqrt_rL * hl    + sqrt_rR * hr)    * inv_den;

    double v2 = u_roe*u_roe + v_roe*v_roe + w_roe*w_roe;
    double c2 = (h_roe - 0.5 * v2) * gamma1;
    if (c2 <= 0.0) c2 = 1.0e-6; 
    double c = std::sqrt(c2);

    double U_contra = u_roe*nx + v_roe*ny + w_roe*nz + nt;
    
    double grad_mag = std::sqrt(nx*nx + ny*ny + nz*nz);
    double sml = 1.0e-30;
    grad_mag = std::max(grad_mag, sml);
    
    double inv_mag = 1.0 / grad_mag;
    double nx_n = nx * inv_mag;
    double ny_n = ny * inv_mag;
    double nz_n = nz * inv_mag;

    double c_mag = c * grad_mag;

    double l1 = std::abs(U_contra - c_mag);
    double l4 = std::abs(U_contra);
    double l5 = std::abs(U_contra + c_mag);

    double delta = efix * c_mag; 
    auto entropy_fix = [&](double lam) {
        if (lam < delta) return 0.5 * (lam*lam + delta*delta) / delta;
        return lam;
    };
    l1 = entropy_fix(l1);
    l4 = entropy_fix(l4);
    l5 = entropy_fix(l5);

    double du = dq[1], dv = dq[2], dw = dq[3];
    double dp = dq[4], drho = dq[0];
    
    double rodcta = rho_roe * (du*nx_n + dv*ny_n + dw*nz_n);
    double dp_c2 = dp / c2;
    double rodctac = rodcta / c;
    
    double a1_coeff = l4 * (drho - dp_c2);          
    double a2_coeff = l5 * (dp_c2 + rodctac) * 0.5; 
    double a3_coeff = l1 * (dp_c2 - rodctac) * 0.5; 
    
    double a4 = a1_coeff + a2_coeff + a3_coeff;
    double a5 = c * (a2_coeff - a3_coeff);
    
    double a6 = l4 * (rho_roe * du - nx_n * rodcta);
    double a7 = l4 * (rho_roe * dv - ny_n * rodcta);
    double a8 = l4 * (rho_roe * dw - nz_n * rodcta);
    
    std::vector<double> df(nl);
    df[0] = a4;
    df[1] = u_roe * a4 + nx_n * a5 + a6;
    df[2] = v_roe * a4 + ny_n * a5 + a7;
    df[3] = w_roe * a4 + nz_n * a5 + a8;
    
    double cta1 = (u_roe*nx_n + v_roe*ny_n + w_roe*nz_n) + nt * inv_mag; 
    df[4] = h_roe * a4 + cta1 * a5 + u_roe*a6 + v_roe*a7 + w_roe*a8 - c2*a1_coeff/gamma1;

    for(int m=0; m<nl; ++m) {
        f[m] = 0.5 * (fl[m] + fr[m] - df[m]);
    }
}

// ===========================================================================
// 辅助：Metrics 插值
// ===========================================================================
static void interp_metrics_half(int n, int ni, const std::vector<double>& q, 
                                std::vector<double>& val)
{
    const double A1=9.0, B1=-1.0;
    const double A2=5.0, B2=15.0, C2=-5.0, D2=1.0;
    const double A3=35.0, B3=-35.0, C3=21.0, D3=-5.0;
    const double dd16=16.0;

    if (ni <= 0) return;

    if (ni <= 2) {
        for (int i = 0; i <= ni; ++i) {
            for (int m = 0; m < n; ++m) 
                val[i*n+m] = 0.5 * (q[(i+2)*n+m] + q[(i+3)*n+m]);
        }
        return;
    }

    for (int m = 0; m < n; ++m) {
        double q1 = q[3*n+m], q2 = q[4*n+m], q3 = q[5*n+m], q4 = q[6*n+m];
        val[0*n+m] = (A3*q1 + B3*q2 + C3*q3 + D3*q4) / dd16;
        val[1*n+m] = (A2*q1 + B2*q2 + C2*q3 + D2*q4) / dd16;
    }

    for (int m = 0; m < n; ++m) {
        double qn   = q[(ni+2)*n+m];
        double qn_1 = q[(ni+1)*n+m];
        double qn_2 = q[(ni+0)*n+m];
        double qn_3 = q[(ni-1)*n+m];
        val[ni*n+m]     = (A3*qn + B3*qn_1 + C3*qn_2 + D3*qn_3) / dd16;
        val[(ni-1)*n+m] = (A2*qn + B2*qn_1 + C2*qn_2 + D2*qn_3) / dd16;
    }

    for (int i = 2; i <= ni-2; ++i) {
        for (int m = 0; m < n; ++m) {
            double q_i   = q[(i+2)*n+m];
            double q_ip1 = q[(i+3)*n+m];
            double q_ip2 = q[(i+4)*n+m];
            double q_im1 = q[(i+1)*n+m];
            val[i*n+m] = (A1*(q_i + q_ip1) + B1*(q_ip2 + q_im1)) / dd16;
        }
    }
}

// ===========================================================================
// WCNS 重构 + 通量计算
// ===========================================================================
void InviscidFluxComputer::compute_flux_line(int ni, 
                                             const std::vector<double>& q_line,
                                             const std::vector<double>& met_line,
                                             std::vector<double>& fc,
                                             const orion::core::Params& params)
{
    if (ni <= 0) return;

    int nl = 5;
    double efix = 0.1;
    double gamma = params.inflow.gama;
    double small = 1.0e-20;

    int n_face = ni + 1;
    // [Safe Alloc]
    if (n_face <= 0 || n_face > 1000000) return; // Sanity check

    std::vector<double> met_half(n_face * 5);
    interp_metrics_half(5, ni, met_line, met_half);

    std::vector<double> qwl(n_face * nl), qwr(n_face * nl);
    const double CL1=1.0/16.0, CL2=10.0/16.0, CL3=5.0/16.0;
    const double EPS=1.0e-6;

    int ist = 0, ied = ni;
    if (q_line[0*6 + 0] < small) ist = 2; 
    if (q_line[(ni+5)*6 + 0] < small) ied = ni - 2;

    for (int m = 0; m < nl; ++m) { 
        for (int i = ist; i <= ied; ++i) { 
            double q_m2 = q_line[(i+0)*6 + m];
            double q_m1 = q_line[(i+1)*6 + m];
            double q_0  = q_line[(i+2)*6 + m];
            double q_p1 = q_line[(i+3)*6 + m];
            double q_p2 = q_line[(i+4)*6 + m];

            double s1 = q_m2 - 2.0*q_m1 + q_0;
            double s2 = q_m1 - 2.0*q_0  + q_p1;
            double s3 = q_0  - 2.0*q_p1 + q_p2;

            double g1 = 0.5 * (q_m2 - 4.0*q_m1 + 3.0*q_0);
            double g2 = 0.5 * (q_p1 - q_m1);
            double g3 = 0.5 * (-3.0*q_0 + 4.0*q_p1 - q_p2);

            auto calc_w = [&](double g, double s, double C) {
                double IS = g*g + s*s;
                return C / ((EPS + IS)*(EPS + IS));
            };

            double wl1 = calc_w(g1, s1, CL1);
            double wl2 = calc_w(g2, s2, CL2);
            double wl3 = calc_w(g3, s3, CL3);
            double sum_wl = wl1 + wl2 + wl3;
            qwl[i*nl+m] = q_0 + 0.125*((wl1/sum_wl)*(s1+4.0*g1) + (wl2/sum_wl)*(s2+4.0*g2) + (wl3/sum_wl)*(s3+4.0*g3));

            double sr1 = s2; 
            double sr2 = s3; 
            double sr3 = q_p1 - 2.0*q_p2 + q_line[(i+5)*6+m];

            double qr_0 = q_p1; 
            double qr_m1 = q_0;
            double qr_m2 = q_m1;
            double qr_p1 = q_p2;
            double qr_p2 = q_line[(i+5)*6+m];

            double gr1 = 0.5 * (qr_m2 - 4.0*qr_m1 + 3.0*qr_0);
            double gr2 = 0.5 * (qr_p1 - qr_m1);
            double gr3 = 0.5 * (-3.0*qr_0 + 4.0*qr_p1 - qr_p2);
            
            double wr1 = calc_w(gr1, sr1, CL3); 
            double wr2 = calc_w(gr2, sr2, CL2); 
            double wr3 = calc_w(gr3, sr3, CL1); 
            double sum_wr = wr1 + wr2 + wr3;
            qwr[i*nl+m] = q_p1 + 0.125*((wr1/sum_wr)*(sr1-4.0*gr1) + (wr2/sum_wr)*(sr2-4.0*gr2) + (wr3/sum_wr)*(sr3-4.0*gr3));
        }
    }
    
    if (ist > 1) {
        for(int m=0; m<nl; ++m) {
            double q1 = q_line[3*6+m], q2 = q_line[4*6+m], q3 = q_line[5*6+m], q4 = q_line[6*6+m];
            qwl[0*nl+m] = (35.*q1 - 35.*q2 + 21.*q3 - 5.*q4)/16.;
            qwl[1*nl+m] = (5.*q1 + 15.*q2 - 5.*q3 + q4)/16.;
            qwr[0*nl+m] = qwl[0*nl+m];
            qwr[1*nl+m] = qwl[1*nl+m];
        }
    }
    if (ied < ni) {
        for(int m=0; m<nl; ++m) {
            double qn = q_line[(ni+2)*6+m], qn1 = q_line[(ni+1)*6+m], qn2 = q_line[(ni)*6+m], qn3 = q_line[(ni-1)*6+m];
            qwr[ni*nl+m] = (35.*qn - 35.*qn1 + 21.*qn2 - 5.*qn3)/16.;
            qwr[(ni-1)*nl+m] = (5.*qn + 15.*qn1 - 5.*qn2 + qn3)/16.;
            qwl[ni*nl+m] = qwr[ni*nl+m];
            qwl[(ni-1)*nl+m] = qwr[(ni-1)*nl+m];
        }
    }

    std::vector<double> f(n_face * nl);
    std::vector<double> ql_loc(nl), qr_loc(nl), f_loc(nl);

    for (int i = 0; i <= ni; ++i) {
        for(int m=0; m<nl; ++m) { ql_loc[m] = qwl[i*nl+m]; qr_loc[m] = qwr[i*nl+m]; }
        
        double r_min = 1e-6, p_min = 1e-6;
        if(ql_loc[0] < r_min || ql_loc[4] < p_min) {
            int idx = std::max(0, i-1) + 2; 
            for(int m=0; m<nl; ++m) ql_loc[m] = q_line[idx*6+m];
        }
        if(qr_loc[0] < r_min || qr_loc[4] < p_min) {
            int idx = std::min(ni-1, i) + 2 + 1; 
            for(int m=0; m<nl; ++m) qr_loc[m] = q_line[idx*6+m];
        }

        flux_Roe(ql_loc, qr_loc, met_half[i*5+0], met_half[i*5+1], met_half[i*5+2], met_half[i*5+3], efix, gamma, f_loc);
        for(int m=0; m<nl; ++m) f[i*nl+m] = f_loc[m];
    }

    for (int i = 0; i < ni; ++i) { 
        for(int m=0; m<nl; ++m) {
            fc[i*nl+m] = f[(i+1)*nl+m] - f[i*nl+m];
        }
    }

    if (ni >= 5) {
        if (ist > 1) { 
            for (int m = 0; m < nl; ++m) {
                fc[0*nl+m] = (-23.0*f[0*nl+m] + 21.0*f[1*nl+m] + 3.0*f[2*nl+m] - 1.0*f[3*nl+m]) / 24.0;
                fc[1*nl+m] = (-22.0*f[1*nl+m] + 17.0*f[2*nl+m] + 9.0*f[3*nl+m] - 5.0*f[4*nl+m] + f[5*nl+m]) / 24.0;
                fc[2*nl+m] = (f[1*nl+m] - 27.0*f[2*nl+m] + 27.0*f[3*nl+m] - f[4*nl+m]) / 24.0;
            }
        }

        if (ied < ni) { 
            int N = ni; 
            for (int m = 0; m < nl; ++m) {
                fc[(ni-1)*nl+m] = -(-23.0*f[N*nl+m] + 21.0*f[(N-1)*nl+m] + 3.0*f[(N-2)*nl+m] - 1.0*f[(N-3)*nl+m]) / 24.0;
                fc[(ni-2)*nl+m] = -(-22.0*f[(N-1)*nl+m] + 17.0*f[(N-2)*nl+m] + 9.0*f[(N-3)*nl+m] - 5.0*f[(N-4)*nl+m] + f[(N-5)*nl+m]) / 24.0;
                fc[(ni-3)*nl+m] = -(f[(N-1)*nl+m] - 27.0*f[(N-2)*nl+m] + 27.0*f[(N-3)*nl+m] - f[(N-4)*nl+m]) / 24.0;
            }
        }
    }
}

void InviscidFluxComputer::compute_inviscid_rhs(orion::preprocess::FlowFieldSet& fs, 
                                                const orion::core::Params& params)
{
    // [Global Try-Catch] 捕获任何分配失败，并打印具体信息
    try {
        int ng = fs.ng;
        if (ng <= 0) ng = 1; // Sanity check

        for (int nb : fs.local_block_ids) {
            compute_block_inviscid(fs.blocks[nb], params, ng);
        }
    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR CAUGHT in compute_inviscid_rhs]: " << e.what() << std::endl;
        std::exit(-1);
    } catch (...) {
        std::cerr << "[FATAL ERROR CAUGHT in compute_inviscid_rhs]: Unknown exception." << std::endl;
        std::exit(-1);
    }
}

void InviscidFluxComputer::compute_block_inviscid(orion::preprocess::BlockField& bf,
                                                  const orion::core::Params& params,
                                                  int ng)
{
    const auto& dims = bf.prim.dims();
    if (dims.size() < 3) return;

    int nx = dims[0], ny = dims[1], nz = dims[2];
    int idim = nx - 2*ng, jdim = ny - 2*ng, kdim = nz - 2*ng;
    
    // 获取变量数 (如果 dq 分配了 5个变量，就用5)
    int nl = 5; 
    if (bf.dq.dims().size() > 3) nl = bf.dq.dims()[3];

    // [关键] 检查 dimensions 是否足以支撑计算
    if (idim <= 0 || jdim <= 0 || kdim <= 0) {
        // Degenerated block for inviscid calculation (too small)
        return; 
    }

    int max_dim = std::max({idim, jdim, kdim});
    // line_size 计算
    long long line_size = (long long)max_dim + 2*ng + 6;
    
    // [Explicit Check] 防止 line_size 异常大
    if (line_size > 10000000) { 
        std::cerr << "[ERROR] Absurd line_size: " << line_size << " (nx="<<nx<<", ng="<<ng<<")\n";
        return;
    }

    // 预分配内存 (使用 vector resize)
    std::vector<double> q_line; q_line.resize(line_size * 6); 
    std::vector<double> met_line; met_line.resize(line_size * 5); 
    std::vector<double> fc; fc.resize(line_size * nl); 

    // --- I Direction ---
    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            int st_f = -2, ed_f = idim + 3;
            for (int i_f = st_f; i_f <= ed_f; ++i_f) {
                int idx = std::max(0, std::min(nx-1, ng + i_f - 1));
                int line_idx = i_f - st_f;
                if (line_idx < 0 || line_idx * 6 + 5 >= (int)q_line.size()) continue; // Bounds check

                for (int m = 0; m < 5; ++m) q_line[line_idx * 6 + m] = bf.prim(idx, j, k, m);
                q_line[line_idx * 6 + 5] = bf.T(idx, j, k);

                met_line[line_idx * 5 + 0] = bf.metrics(idx, j, k, 0); 
                met_line[line_idx * 5 + 1] = bf.metrics(idx, j, k, 1); 
                met_line[line_idx * 5 + 2] = bf.metrics(idx, j, k, 2); 
                met_line[line_idx * 5 + 3] = 0.0;
                met_line[line_idx * 5 + 4] = bf.vol(idx, j, k);
            }
            compute_flux_line(idim, q_line, met_line, fc, params);
            for (int i_f = 1; i_f <= idim; ++i_f) {
                int idx = ng + i_f - 1;
                for (int m = 0; m < nl; ++m) bf.dq(idx, j, k, m) += fc[(i_f-1) * nl + m];
            }
        }
    }

    // --- J Direction ---
    for (int k = ng; k < nz - ng; ++k) {
        for (int i = ng; i < nx - ng; ++i) {
            int st_f = -2, ed_f = jdim + 3;
            for (int j_f = st_f; j_f <= ed_f; ++j_f) {
                int idx = std::max(0, std::min(ny-1, ng + j_f - 1));
                int line_idx = j_f - st_f;
                if (line_idx < 0 || line_idx * 6 + 5 >= (int)q_line.size()) continue;

                for (int m = 0; m < 5; ++m) q_line[line_idx * 6 + m] = bf.prim(i, idx, k, m);
                q_line[line_idx * 6 + 5] = bf.T(i, idx, k);

                met_line[line_idx * 5 + 0] = bf.metrics(i, idx, k, 3);
                met_line[line_idx * 5 + 1] = bf.metrics(i, idx, k, 4);
                met_line[line_idx * 5 + 2] = bf.metrics(i, idx, k, 5);
                met_line[line_idx * 5 + 3] = 0.0;
                met_line[line_idx * 5 + 4] = bf.vol(i, idx, k);
            }
            compute_flux_line(jdim, q_line, met_line, fc, params);
            for (int j_f = 1; j_f <= jdim; ++j_f) {
                int idx = ng + j_f - 1;
                for (int m = 0; m < nl; ++m) bf.dq(i, idx, k, m) += fc[(j_f-1) * nl + m];
            }
        }
    }

    // --- K Direction ---
    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            int st_f = -2, ed_f = kdim + 3;
            for (int k_f = st_f; k_f <= ed_f; ++k_f) {
                int idx = std::max(0, std::min(nz-1, ng + k_f - 1));
                int line_idx = k_f - st_f;
                if (line_idx < 0 || line_idx * 6 + 5 >= (int)q_line.size()) continue;

                for (int m = 0; m < 5; ++m) q_line[line_idx * 6 + m] = bf.prim(i, j, idx, m);
                q_line[line_idx * 6 + 5] = bf.T(i, j, idx);

                met_line[line_idx * 5 + 0] = bf.metrics(i, j, idx, 6);
                met_line[line_idx * 5 + 1] = bf.metrics(i, j, idx, 7);
                met_line[line_idx * 5 + 2] = bf.metrics(i, j, idx, 8);
                met_line[line_idx * 5 + 3] = 0.0;
                met_line[line_idx * 5 + 4] = bf.vol(i, j, idx);
            }
            compute_flux_line(kdim, q_line, met_line, fc, params);
            for (int k_f = 1; k_f <= kdim; ++k_f) {
                int idx = ng + k_f - 1;
                for (int m = 0; m < nl; ++m) bf.dq(i, j, idx, m) += fc[(k_f-1) * nl + m];
            }
        }
    }
}

} // namespace orion::solver