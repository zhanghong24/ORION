#include "solver/StateUpdater.hpp"
#include "solver/HaloExchanger.hpp"
#include "core/Runtime.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>

namespace orion::solver {

// ===========================================================================
// PART 1: Auxiliary Variables Update (Recast Field)
// ===========================================================================

static void update_thermodynamics_block(orion::preprocess::BlockField& bf, 
                                        double gamma, double moo, int nvis, int ng)
{
    const auto& dims = bf.prim.dims();
    int i_alloc = dims[0];
    int j_alloc = dims[1];
    int k_alloc = dims[2];

    int idim = i_alloc - 2 * ng;
    int jdim = j_alloc - 2 * ng;
    int kdim = k_alloc - 2 * ng;

    double moo2 = moo * moo;

    // 1. 更新内部区域 (Inner)
    int i_s = ng, i_e = ng + idim - 1;
    int j_s = ng, j_e = ng + jdim - 1;
    int k_s = ng, k_e = ng + kdim - 1;

    for (int k = k_s; k <= k_e; ++k) {
        for (int j = j_s; j <= j_e; ++j) {
            for (int i = i_s; i <= i_e; ++i) {
                double rho = bf.prim(i, j, k, 0);
                double p   = bf.prim(i, j, k, 4);
                // 内部区域假定 rho > 0
                double a2 = gamma * p / rho;
                bf.c(i, j, k) = std::sqrt(std::abs(a2));
                
                // [修正] T 存储在 prim(..., 5)
                if (nvis == 1) bf.prim(i, j, k, 5) = moo2 * a2;
            }
        }
    }

    // 2. 更新 Ghost 区域 (处理可能的负密度)
    int layers_to_update = ng; // 覆盖所有 Ghost 以策安全

    for (int dir = 0; dir < 3; ++dir) {
        for (int l = 1; l <= layers_to_update; ++l) {
            int idx_min = ng - l;
            int idx_max = ng + (dir==0?idim:dir==1?jdim:kdim) - 1 + l;
            
            int faces[2] = {idx_min, idx_max};
            for (int f_idx : faces) {
                int k_start = (dir==2) ? f_idx : k_s;
                int k_stop  = (dir==2) ? f_idx : k_e;
                int j_start = (dir==1) ? f_idx : j_s;
                int j_stop  = (dir==1) ? f_idx : j_e;
                int i_start = (dir==0) ? f_idx : i_s;
                int i_stop  = (dir==0) ? f_idx : i_e;

                for (int k = k_start; k <= k_stop; ++k) {
                    for (int j = j_start; j <= j_stop; ++j) {
                        for (int i = i_start; i <= i_stop; ++i) {
                            double rho = bf.prim(i, j, k, 0);
                            double p   = bf.prim(i, j, k, 4);
                            
                            // Ghost 区域取绝对值密度，防止 sqrt 崩溃
                            rho = std::abs(rho);
                            
                            double a2 = gamma * p / rho;
                            bf.c(i, j, k) = std::sqrt(std::abs(a2));
                            
                            // [修正] T 存储在 prim(..., 5)
                            if (nvis == 1) bf.prim(i, j, k, 5) = moo2 * a2;
                        }
                    }
                }
            }
        }
    }
}

static void update_viscosity_block(orion::preprocess::BlockField& bf, 
                                   double visc, int method, int ng)
{
    const auto& dims = bf.prim.dims();
    
    // [策略说明] 
    // 虽然 Fortran compute_visl_ns 循环看起来从 1 开始，
    // 但 newvis.f90 的 Flux 计算使用了 i=0 的粘性。
    // 为防止边界通量错误，C++ 这里计算全场（包括 Ghost）。
    int i_s = 0, i_e = dims[0] - 1;
    int j_s = 0, j_e = dims[1] - 1;
    int k_s = 0, k_e = dims[2] - 1;

    for (int k = k_s; k <= k_e; ++k) {
        for (int j = j_s; j <= j_e; ++j) {
            for (int i = i_s; i <= i_e; ++i) {
                // [修正] T 从 prim(..., 5) 读取
                double tm = bf.prim(i, j, k, 5); 
                
                if (tm < 1e-6) tm = 1e-6; // 保护防止除零
                bf.mu(i, j, k) = tm * std::sqrt(tm) * (1.0 + visc) / (tm + visc);
            }
        }
    }
}

static void update_conservative_block(orion::preprocess::BlockField& bf, 
                                      double gamma, int ng)
{
    const auto& dims = bf.prim.dims();
    int idim = dims[0] - 2 * ng;
    int jdim = dims[1] - 2 * ng;
    int kdim = dims[2] - 2 * ng;

    double gm1_inv = 1.0 / (gamma - 1.0);

    int i_s = ng, i_e = ng + idim - 1;
    int j_s = ng, j_e = ng + jdim - 1;
    int k_s = ng, k_e = ng + kdim - 1;

    auto calc_q = [&](int i, int j, int k, bool use_abs_rho) {
        double rho = bf.prim(i, j, k, 0);
        double u   = bf.prim(i, j, k, 1);
        double v   = bf.prim(i, j, k, 2);
        double w   = bf.prim(i, j, k, 3);
        double p   = bf.prim(i, j, k, 4);

        // [复刻 Fortran] Ghost 区域强制使用正密度计算 Q
        double rho_calc = (use_abs_rho || rho < 0.0) ? std::abs(rho) : rho;

        bf.q(i, j, k, 0) = rho_calc; 
        bf.q(i, j, k, 1) = rho_calc * u;
        bf.q(i, j, k, 2) = rho_calc * v;
        bf.q(i, j, k, 3) = rho_calc * w;

        double ke = 0.5 * rho_calc * (u*u + v*v + w*w);
        bf.q(i, j, k, 4) = p * gm1_inv + ke;
    };

    // 1. 更新内部区域
    for (int k = k_s; k <= k_e; ++k) 
        for (int j = j_s; j <= j_e; ++j) 
            for (int i = i_s; i <= i_e; ++i) 
                calc_q(i, j, k, false);

    // 2. 更新所有 Ghost 区域
    int layers_to_update = ng; 
    
    for (int dir = 0; dir < 3; ++dir) {
        for (int l = 1; l <= layers_to_update; ++l) {
            int idx_min = ng - l;
            int idx_max = ng + (dir==0?idim:dir==1?jdim:kdim) - 1 + l;
            int faces[2] = {idx_min, idx_max};
            for (int f_idx : faces) {
                int k_st = (dir==2) ? f_idx : k_s; int k_ed = (dir==2) ? f_idx : k_e;
                int j_st = (dir==1) ? f_idx : j_s; int j_ed = (dir==1) ? f_idx : j_e;
                int i_st = (dir==0) ? f_idx : i_s; int i_ed = (dir==0) ? f_idx : i_e;
                
                for (int k = k_st; k <= k_ed; ++k)
                    for (int j = j_st; j <= j_ed; ++j)
                        for (int i = i_st; i <= i_ed; ++i)
                            calc_q(i, j, k, true);
            }
        }
    }
}

void StateUpdater::update_flow_states(orion::preprocess::FlowFieldSet& fs, 
                                      const orion::core::Params& params)
{
    double gamma = params.inflow.gama;
    double moo = params.inflow.moo;
    double visc = params.inflow.visc;
    int nvis = params.flowtype.nvis;
    int method = params.control.method;
    int ng = fs.ng;

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        
        // 1. 更新热力学变量 (c, T)
        update_thermodynamics_block(bf, gamma, moo, nvis, ng);
        
        // 2. 更新粘性系数 (mu)
        if (nvis == 1) {
            update_viscosity_block(bf, visc, method, ng);
        }
        
        // 3. 更新守恒变量 (Q)
        update_conservative_block(bf, gamma, ng);
        
        // 4. 清零残差 (DQ)
        bf.dq.fill(0.0);
    }
}


// ===========================================================================
// PART 2: Implicit Solver & Solution Update (LU-SGS)
// ===========================================================================

static double calc_spectral_radius(const double* prim, double gamma, 
                                   double kx, double ky, double kz, double kt) 
{
    double rho = prim[0];
    double u = prim[1], v = prim[2], w = prim[3];
    double p = prim[4];
    
    // 保护
    if (rho < 1e-30) rho = 1e-30;
    if (p < 1e-30) p = 1e-30;

    double c2 = gamma * p / rho;
    double c = std::sqrt(c2);
    
    double U = u*kx + v*ky + w*kz + kt;
    double grad_mag = std::sqrt(kx*kx + ky*ky + kz*kz);
    return std::abs(U) + c * grad_mag;
}

static double calc_visc_spectral_radius(const double* prim, double mu, double kx, double ky, double kz) {
    double rho = prim[0];
    if (rho < 1e-30) rho = 1e-30;
    double grad2 = kx*kx + ky*ky + kz*kz;
    return (mu / rho) * grad2; 
}

static void compute_flux_splitting(const double* prim, 
                                   const double* dq,
                                   const double* metric,
                                   double spec_radius, 
                                   int sign,
                                   double gamma,
                                   double* result) 
{
    double rm = prim[0], um = prim[1], vm = prim[2], wm = prim[3], pm = prim[4];
    double drm = dq[0], dum = dq[1], dvm = dq[2], dwm = dq[3], dEm = dq[4]; 

    double nx = metric[0], ny = metric[1], nz = metric[2], nt = metric[3];
    double v2 = um*um + vm*vm + wm*wm;
    
    if (rm < 1e-30) rm = 1e-30;
    if (pm < 1e-30) pm = 1e-30;

    double c2 = gamma * pm / rm;
    double cm = std::sqrt(c2);
    double H = (gamma / (gamma - 1.0)) * (pm / rm) + 0.5 * v2; 

    double ct = nx*um + ny*vm + nz*wm + nt;
    double grad_mag = std::sqrt(nx*nx + ny*ny + nz*nz);
    double sml = 1.0e-30;
    if (grad_mag < sml) grad_mag = sml;
    
    // Splitting
    double l1 = ct;
    double l4 = ct + cm * grad_mag;
    double l5 = ct - cm * grad_mag;
    
    l1 = 0.5 * (l1 + sign * spec_radius);
    l4 = 0.5 * (l4 + sign * spec_radius);
    l5 = 0.5 * (l5 + sign * spec_radius);
    
    double x1 = (2.0*l1 - l4 - l5) / (2.0 * c2);
    double x2 = (l4 - l5) / (2.0 * cm); 
    
    double inv_mag = 1.0 / grad_mag;
    double nx_n = nx * inv_mag;
    double ny_n = ny * inv_mag;
    double nz_n = nz * inv_mag;
    double ct_n = (ct - nt) * inv_mag;
    
    double gm1 = gamma - 1.0;
    double af = 0.5 * gm1 * v2;
    
    double dc = ct_n * drm - (nx_n * dum + ny_n * dvm + nz_n * dwm);
    double dh = af * drm - gm1 * (um*dum + vm*dvm + wm*dwm - dEm);
    
    double c2dc = c2 * dc;
    
    result[0] = l1 * drm - dh * x1 - dc * x2;
    result[1] = l1 * dum + (nx_n*c2dc - um*dh)*x1 + (nx_n*dh - um*dc)*x2;
    result[2] = l1 * dvm + (ny_n*c2dc - vm*dh)*x1 + (ny_n*dh - vm*dc)*x2;
    result[3] = l1 * dwm + (nz_n*c2dc - wm*dh)*x1 + (nz_n*dh - wm*dc)*x2;
    result[4] = l1 * dEm + (ct_n*c2dc - H*dh)*x1 + (ct_n*dh - H*dc)*x2;
}

static double get_diag_coeff(int i, int j, int k, 
                             orion::preprocess::BlockField& bf,
                             double gamma,
                             double dt_local,
                             int nvis)
{
    double prim[5];
    for(int m=0; m<5; ++m) prim[m] = bf.prim(i,j,k,m);
    
    double ra = calc_spectral_radius(prim, gamma, bf.metrics(i,j,k,0), bf.metrics(i,j,k,1), bf.metrics(i,j,k,2), 0.0);
    double rb = calc_spectral_radius(prim, gamma, bf.metrics(i,j,k,3), bf.metrics(i,j,k,4), bf.metrics(i,j,k,5), 0.0);
    double rc = calc_spectral_radius(prim, gamma, bf.metrics(i,j,k,6), bf.metrics(i,j,k,7), bf.metrics(i,j,k,8), 0.0);
    
    double rva=0, rvb=0, rvc=0;
    if (nvis == 1) {
        double mu = bf.mu(i,j,k);
        rva = calc_visc_spectral_radius(prim, mu, bf.metrics(i,j,k,0), bf.metrics(i,j,k,1), bf.metrics(i,j,k,2));
        rvb = calc_visc_spectral_radius(prim, mu, bf.metrics(i,j,k,3), bf.metrics(i,j,k,4), bf.metrics(i,j,k,5));
        rvc = calc_visc_spectral_radius(prim, mu, bf.metrics(i,j,k,6), bf.metrics(i,j,k,7), bf.metrics(i,j,k,8));
    }
    
    double rad_sum = ra + rb + rc + rva + rvb + rvc;
    
    double beta = 1.0; 
    double wmig = 1.0;
    double term = (1.0/dt_local) + beta * wmig * rad_sum;
    return 1.0 / term; 
}

static void lusgs_sweep(orion::preprocess::BlockField& bf,
                        const std::vector<double>& rhs_vec,
                        double gamma, int nvis, int ng,
                        int dir) 
{
    int nx = bf.dq.dims()[0];
    int ny = bf.dq.dims()[1];
    int nz = bf.dq.dims()[2];
    int nl = 5;

    int ist = (dir == 1) ? ng : nx - ng - 1;
    int ied = (dir == 1) ? nx - ng : ng - 1;
    int jst = (dir == 1) ? ng : ny - ng - 1;
    int jed = (dir == 1) ? ny - ng : ng - 1;
    int kst = (dir == 1) ? ng : nz - ng - 1;
    int ked = (dir == 1) ? nz - ng : ng - 1;

    std::vector<double> prim_nb(nl), dq_nb(nl), flux_split(nl), rhs0(nl);
    
    auto loop_cond = [dir](int c, int e) { return (dir==1) ? (c < e) : (c > e); };

    for (int k = kst; loop_cond(k, ked); k += dir) {
        for (int j = jst; loop_cond(j, jed); j += dir) {
            for (int i = ist; loop_cond(i, ied); i += dir) {
                
                double dt = bf.dt(i,j,k);
                double coed = get_diag_coeff(i,j,k, bf, gamma, dt, nvis);
                
                std::fill(rhs0.begin(), rhs0.end(), 0.0);

                auto add_nb = [&](int in, int jn, int kn, int m_idx) {
                    for(int m=0; m<nl; ++m) { prim_nb[m]=bf.prim(in,jn,kn,m); dq_nb[m]=bf.dq(in,jn,kn,m); }
                    double metric[4] = {bf.metrics(i,j,k,m_idx), bf.metrics(i,j,k,m_idx+1), bf.metrics(i,j,k,m_idx+2), 0.0};
                    double r = calc_spectral_radius(prim_nb.data(), gamma, metric[0], metric[1], metric[2], 0);
                    compute_flux_splitting(prim_nb.data(), dq_nb.data(), metric, r, dir, gamma, flux_split.data());
                    for(int m=0; m<nl; ++m) rhs0[m] += flux_split[m];
                };

                add_nb(i-dir, j, k, 0); 
                add_nb(i, j-dir, k, 3); 
                add_nb(i, j, k-dir, 6); 

                long long idx = ((long long)k * ny + j) * nx + i;
                double wmig = 1.0;
                
                for(int m=0; m<nl; ++m) {
                    double explicit_term = -rhs_vec[idx*nl + m]; 
                    bf.dq(i,j,k,m) = (explicit_term + wmig * rhs0[m]) * coed;
                }
            }
        }
    }
}

static void gs_pr_sweep(orion::preprocess::BlockField& bf,
                        const std::vector<double>& rhs_vec,
                        double gamma, int nvis, int ng,
                        int dir)
{
    int nx = bf.dq.dims()[0];
    int ny = bf.dq.dims()[1];
    int nz = bf.dq.dims()[2];
    int nl = 5;

    int ist = (dir == 1) ? ng : nx - ng - 1;
    int ied = (dir == 1) ? nx - ng : ng - 1;
    int jst = (dir == 1) ? ng : ny - ng - 1;
    int jed = (dir == 1) ? ny - ng : ng - 1;
    int kst = (dir == 1) ? ng : nz - ng - 1;
    int ked = (dir == 1) ? nz - ng : ng - 1;

    std::vector<double> prim_nb(nl), dq_nb(nl), flux_split(nl), rhs0(nl);
    auto loop_cond = [dir](int c, int e) { return (dir==1) ? (c < e) : (c > e); };

    for (int k = kst; loop_cond(k, ked); k += dir) {
        for (int j = jst; loop_cond(j, jed); j += dir) {
            for (int i = ist; loop_cond(i, ied); i += dir) {
                
                double dt = bf.dt(i,j,k);
                double coed = get_diag_coeff(i,j,k, bf, gamma, dt, nvis);
                std::fill(rhs0.begin(), rhs0.end(), 0.0);

                auto add_nb_contrib = [&](int in, int jn, int kn, int m_idx, int sgn) {
                    for(int m=0; m<nl; ++m) { prim_nb[m]=bf.prim(in,jn,kn,m); dq_nb[m]=bf.dq(in,jn,kn,m); }
                    double metric[4] = {bf.metrics(i,j,k,m_idx), bf.metrics(i,j,k,m_idx+1), bf.metrics(i,j,k,m_idx+2), 0.0};
                    double r = calc_spectral_radius(prim_nb.data(), gamma, metric[0], metric[1], metric[2], 0);
                    compute_flux_splitting(prim_nb.data(), dq_nb.data(), metric, r, sgn, gamma, flux_split.data());
                    if(sgn==1) { for(int m=0; m<nl; ++m) rhs0[m] += flux_split[m]; }
                    else       { for(int m=0; m<nl; ++m) rhs0[m] -= flux_split[m]; }
                };

                // Left Neighbors (A+)
                add_nb_contrib(i-1, j, k, 0, 1);
                add_nb_contrib(i, j-1, k, 3, 1);
                add_nb_contrib(i, j, k-1, 6, 1);

                // Right Neighbors (A-)
                add_nb_contrib(i+1, j, k, 0, -1);
                add_nb_contrib(i, j+1, k, 3, -1);
                add_nb_contrib(i, j, k+1, 6, -1);

                long long idx = ((long long)k * ny + j) * nx + i;
                double wmig = 1.0;
                for(int m=0; m<nl; ++m) {
                    double resid = -rhs_vec[idx*nl+m];
                    bf.dq(i,j,k,m) = (wmig * rhs0[m] + resid) * coed;
                }
            }
        }
    }
}

static void set_boundary_dq_zero(orion::preprocess::BlockField& bf, 
                                 const orion::bc::BlockBC& bcb,
                                 int ng)
{
    for (int nr = 0; nr < bcb.nregions; ++nr) {
        const auto& reg = bcb.regions[nr];
        if (reg.bctype > 0) {
            int is = reg.s_st[0] - 1 + ng;
            int ie = reg.s_ed[0] - 1 + ng;
            int js = reg.s_st[1] - 1 + ng;
            int je = reg.s_ed[1] - 1 + ng;
            int ks = reg.s_st[2] - 1 + ng;
            int ke = reg.s_ed[2] - 1 + ng;
            
            for (int k = ks; k <= ke; ++k)
                for (int j = js; j <= je; ++j)
                    for (int i = is; i <= ie; ++i)
                        for (int m = 0; m < 5; ++m)
                            bf.dq(i, j, k, m) = 0.0;
        }
    }
}

static void update_limited(orion::preprocess::BlockField& bf, 
                           const orion::core::Params& params,
                           int ng)
{
    const auto& dims = bf.dq.dims();
    int nx = dims[0], ny = dims[1], nz = dims[2];
    double gamma = params.inflow.gama;
    double gm1 = gamma - 1.0;

    int ist = ng, ied = nx - ng;
    int jst = ng, jed = ny - ng;
    int kst = ng, ked = nz - ng;

    double p_min = 1e-6, r_min = 1e-6; 

    for (int k = kst; k < ked; ++k) {
        for (int j = jst; j < jed; ++j) {
            for (int i = ist; i < ied; ++i) {
                
                double r = bf.prim(i,j,k,0);
                double u = bf.prim(i,j,k,1);
                double v = bf.prim(i,j,k,2);
                double w = bf.prim(i,j,k,3);
                double p = bf.prim(i,j,k,4);
                
                double E = p/gm1 + 0.5*r*(u*u + v*v + w*w);
                
                double q_new[5];
                q_new[0] = r + bf.dq(i,j,k,0);
                q_new[1] = r*u + bf.dq(i,j,k,1);
                q_new[2] = r*v + bf.dq(i,j,k,2);
                q_new[3] = r*w + bf.dq(i,j,k,3);
                q_new[4] = E + bf.dq(i,j,k,4);
                
                double r_n = q_new[0];
                double inv_r = 1.0/r_n;
                double p_n = gm1 * (q_new[4] - 0.5*r_n*( std::pow(q_new[1]*inv_r,2) + std::pow(q_new[2]*inv_r,2) + std::pow(q_new[3]*inv_r,2) ));
                
                double dp_p = std::abs((p_n - p)/p);
                double term = std::max(0.0, -0.3 + dp_p); 
                double Falpha = 1.0 / (1.0 + 2.0 * term);
                
                double q_final[5];
                q_final[0] = r + bf.dq(i,j,k,0) * Falpha;
                q_final[1] = r*u + bf.dq(i,j,k,1) * Falpha;
                q_final[2] = r*v + bf.dq(i,j,k,2) * Falpha;
                q_final[3] = r*w + bf.dq(i,j,k,3) * Falpha;
                q_final[4] = E + bf.dq(i,j,k,4) * Falpha;
                
                r_n = q_final[0];
                inv_r = 1.0/r_n;
                double u_n = q_final[1]*inv_r;
                double v_n = q_final[2]*inv_r;
                double w_n = q_final[3]*inv_r;
                double p_n_final = gm1 * (q_final[4] - 0.5*r_n*(u_n*u_n + v_n*v_n + w_n*w_n));
                
                if (r_n <= r_min || p_n_final <= p_min) {
                    bf.prim(i,j,k,0) = r;
                    bf.prim(i,j,k,1) = u;
                    bf.prim(i,j,k,2) = v;
                    bf.prim(i,j,k,3) = w;
                    bf.prim(i,j,k,4) = p;
                } else {
                    bf.prim(i,j,k,0) = r_n;
                    bf.prim(i,j,k,1) = u_n;
                    bf.prim(i,j,k,2) = v_n;
                    bf.prim(i,j,k,3) = w_n;
                    bf.prim(i,j,k,4) = p_n_final;
                }
            }
        }
    }
}

void StateUpdater::update_solution(orion::preprocess::FlowFieldSet& fs, 
                                   const orion::bc::BCData& bc,
                                   const orion::core::Params& params)
{
    int ng = fs.ng;
    double gamma = params.inflow.gama;
    int nvis = params.flowtype.nvis;

    for (int nb : fs.local_block_ids) {
        auto& bf = fs.blocks[nb];
        const auto& bcb = bc.block_bc[nb]; // Use passed 'bc'

        // 1. Pre-process Boundary DQ
        set_boundary_dq_zero(bf, bcb, ng);

        // 2. Perform LU-SGS
        const auto& dims = bf.dq.dims();
        long long total_size = dims[0] * dims[1] * dims[2] * 5;
        std::vector<double> rhs_vec(total_size);
        
        // Manual copy: bf.dq -> rhs_vec
        int nx=dims[0], ny=dims[1], nz=dims[2], nl=5;
        for(int k=0; k<nz; ++k)
            for(int j=0; j<ny; ++j)
                for(int i=0; i<nx; ++i)
                    for(int m=0; m<nl; ++m) {
                        long long idx = ((long long)k*ny+j)*nx+i;
                        rhs_vec[idx*nl+m] = bf.dq(i,j,k,m);
                    }

        lusgs_sweep(bf, rhs_vec, gamma, nvis, ng, 1);
        lusgs_sweep(bf, rhs_vec, gamma, nvis, ng, -1);
        gs_pr_sweep(bf, rhs_vec, gamma, nvis, ng, 1);
        gs_pr_sweep(bf, rhs_vec, gamma, nvis, ng, -1);

        // 3. Post-process Boundary DQ
        set_boundary_dq_zero(bf, bcb, ng);

        // 4. Update Primitives
        update_limited(bf, params, ng);
    }
}

} // namespace orion::solver