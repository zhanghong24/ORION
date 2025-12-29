#include "bc/PhysicalBC.hpp"
#include "core/Runtime.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace orion::bc {

// ===========================================================================
// 辅助函数：计算归一化法向量 (逐点 Point-wise)
// ===========================================================================
static void get_normal(const orion::preprocess::BlockField& bf,
                       int i, int j, int k, int s_nd,
                       double& nx, double& ny, double& nz)
{
    int offset = (s_nd - 1) * 3;
    
    double mx = bf.metrics(i, j, k, offset + 0);
    double my = bf.metrics(i, j, k, offset + 1);
    double mz = bf.metrics(i, j, k, offset + 2);

    double len = std::sqrt(mx*mx + my*my + mz*mz);
    const double eps = 1.0e-30;
    if (len < eps) len = eps;

    nx = mx / len;
    ny = my / len;
    nz = mz / len;
}

// ===========================================================================
// 辅助函数：计算面的平均法向量 (用于 Symmetry 3D)
// ===========================================================================
static void get_average_normal(const orion::preprocess::BlockField& bf,
                               int i_st, int i_ed,
                               int j_st, int j_ed,
                               int k_st, int k_ed,
                               int s_nd,
                               double& nx_avg, double& ny_avg, double& nz_avg)
{
    int i_mid = (i_st + i_ed) / 2;
    int j_mid = (j_st + j_ed) / 2;
    int k_mid = (k_st + k_ed) / 2;

    double nx, ny, nz;
    get_normal(bf, i_mid, j_mid, k_mid, s_nd, nx, ny, nz);
    
    nx_avg = nx;
    ny_avg = ny;
    nz_avg = nz;
}

// ===========================================================================
// 辅助函数：dif_average (method=1 专用)
// 作用：边界点值 = 0.5 * (Ghost_1 + Inner_1)
// ===========================================================================
static void apply_dif_average(orion::preprocess::BlockField& bf,
                              int i, int j, int k,
                              int dir, int s_lr, int nprim)
{
    int idx_0[3] = {i, j, k}; // 边界点
    int idx_g[3] = {i, j, k}; // Ghost Layer 1
    int idx_i[3] = {i, j, k}; // Inner Layer 1

    if (s_lr == 1) { // Max Face
        idx_g[dir] += 1;
        idx_i[dir] -= 1;
    } else { // Min Face
        idx_g[dir] -= 1;
        idx_i[dir] += 1;
    }

    for (int m = 0; m < nprim; ++m) {
        double val_g = bf.prim(idx_g[0], idx_g[1], idx_g[2], m);
        double val_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], m);
        bf.prim(idx_0[0], idx_0[1], idx_0[2], m) = 0.5 * (val_g + val_i);
    }
}

// ===========================================================================
// 1. 对称边界 (Symmetry / boundary_3)
// 密度不取反！
// ===========================================================================
static void apply_symmetry_bc(const BCRegion& reg, 
                              orion::preprocess::BlockField& bf,
                              const orion::core::Params& params,
                              int ng)
{
    int i_st = reg.s_st[0] - 1 + ng;
    int i_ed = reg.s_ed[0] - 1 + ng;
    int j_st = reg.s_st[1] - 1 + ng;
    int j_ed = reg.s_ed[1] - 1 + ng;
    int k_st = reg.s_st[2] - 1 + ng;
    int k_ed = reg.s_ed[2] - 1 + ng;

    int dir = reg.s_nd - 1;

    int phys_dim = bf.prim.dims()[dir] - 2 * ng;
    bool is_2d = (phys_dim <= 2);

    if (is_2d) {
        // --- 2D Case (展向对称) ---
        for (int k = k_st; k <= k_ed; ++k) {
            for (int j = j_st; j <= j_ed; ++j) {
                for (int i = i_st; i <= i_ed; ++i) {
                    
                    for (int l = 1; l <= ng; ++l) {
                        int idx_g[3] = {i, j, k};
                        int idx_i[3] = {i, j, k};
                        if (reg.s_lr == 1) { idx_g[dir]+=l; idx_i[dir]-=l; }
                        else               { idx_g[dir]-=l; idx_i[dir]+=l; }

                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 0); // Rho Positive
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 1);
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 2);
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 4);
                        if (bf.prim.dims()[3] > 5) 
                             bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 5);

                        double w_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 3);
                        if (l == 1)      bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = -w_i;
                        else             bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = 0.0;
                    }
                    
                    if (params.control.method == 1) 
                        apply_dif_average(bf, i, j, k, dir, reg.s_lr, bf.prim.dims()[3]);
                }
            }
        }
    } else {
        // --- 3D Case (Planar Symmetry) ---
        double nx, ny, nz;
        get_average_normal(bf, i_st, i_ed, j_st, j_ed, k_st, k_ed, reg.s_nd, nx, ny, nz);

        for (int k = k_st; k <= k_ed; ++k) {
            for (int j = j_st; j <= j_ed; ++j) {
                for (int i = i_st; i <= i_ed; ++i) {
                    
                    for (int l = 1; l <= ng; ++l) {
                        int idx_g[3] = {i,j,k};
                        int idx_i[3] = {i,j,k};
                        if (reg.s_lr == 1) { idx_g[dir]+=l; idx_i[dir]-=l; }
                        else               { idx_g[dir]-=l; idx_i[dir]+=l; }

                        double rho = bf.prim(idx_i[0], idx_i[1], idx_i[2], 0);
                        double u   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 1);
                        double v   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 2);
                        double w   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 3);
                        double p   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 4);

                        double v_dot_n = u*nx + v*ny + w*nz;

                        // [修正] 对称边界密度保持正值
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = rho;
                        
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = u - 2.0 * v_dot_n * nx;
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = v - 2.0 * v_dot_n * ny;
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = w - 2.0 * v_dot_n * nz;
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = p;
                        if (bf.prim.dims()[3] > 5) 
                             bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 5);
                    }
                    
                    if (params.control.method == 1) 
                        apply_dif_average(bf, i, j, k, dir, reg.s_lr, bf.prim.dims()[3]);
                }
            }
        }
    }
}

// ===========================================================================
// 2. 无粘壁面 (Inviscid Wall)
// 密度在最外层取反
// ===========================================================================
static void apply_inviscid_wall(const BCRegion& reg, 
                                orion::preprocess::BlockField& bf,
                                const orion::core::Params& params,
                                int ng)
{
    int i_st = reg.s_st[0] - 1 + ng;
    int i_ed = reg.s_ed[0] - 1 + ng;
    int j_st = reg.s_st[1] - 1 + ng;
    int j_ed = reg.s_ed[1] - 1 + ng;
    int k_st = reg.s_st[2] - 1 + ng;
    int k_ed = reg.s_ed[2] - 1 + ng;

    int dir = reg.s_nd - 1;

    for (int k = k_st; k <= k_ed; ++k) {
        for (int j = j_st; j <= j_ed; ++j) {
            for (int i = i_st; i <= i_ed; ++i) {
                
                double nx, ny, nz;
                get_normal(bf, i, j, k, reg.s_nd, nx, ny, nz);

                for (int l = 1; l <= ng; ++l) {
                    int idx_g[3] = {i, j, k};
                    int idx_i[3] = {i, j, k};

                    if (reg.s_lr == 1) { idx_g[dir]+=l; idx_i[dir]-=l; }
                    else               { idx_g[dir]-=l; idx_i[dir]+=l; }

                    double rho = bf.prim(idx_i[0], idx_i[1], idx_i[2], 0);
                    double u   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 1);
                    double v   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 2);
                    double w   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 3);
                    double p   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 4);
                    double t   = (bf.prim.dims()[3] > 5) ? bf.prim(idx_i[0], idx_i[1], idx_i[2], 5) : 0.0;

                    double v_dot_n = u*nx + v*ny + w*nz;
                    
                    // [TGH] 无粘壁面: 最外层密度取反
                    if (l == ng) bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = -rho;
                    else         bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = rho;
                    
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = u - 2.0 * v_dot_n * nx;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = v - 2.0 * v_dot_n * ny;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = w - 2.0 * v_dot_n * nz;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = p;
                    if (bf.prim.dims()[3] > 5) bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = t;
                }

                if (params.control.method == 1) 
                    apply_dif_average(bf, i, j, k, dir, reg.s_lr, bf.prim.dims()[3]);
            }
        }
    }
}

// ===========================================================================
// 3. 粘性壁面 (Viscous Wall)
// 密度在最外层取反
// ===========================================================================
static void apply_viscid_wall(const BCRegion& reg, 
                              orion::preprocess::BlockField& bf,
                              const orion::core::Params& params,
                              int ng)
{
    int i_st = reg.s_st[0] - 1 + ng;
    int i_ed = reg.s_ed[0] - 1 + ng;
    int j_st = reg.s_st[1] - 1 + ng;
    int j_ed = reg.s_ed[1] - 1 + ng;
    int k_st = reg.s_st[2] - 1 + ng;
    int k_ed = reg.s_ed[2] - 1 + ng;

    int dir = reg.s_nd - 1;

    const auto& F = params.inflow;
    double gama  = F.gama;
    double mach  = F.moo;
    double twall = F.twall; 
    double tref  = F.tref;
    double moocp = mach * mach * gama;
    
    bool isothermal = (twall > 1.0e-6);
    double twall_dim = (isothermal && std::abs(tref)>1e-16) ? twall/tref : twall;

    for (int k = k_st; k <= k_ed; ++k) {
        for (int j = j_st; j <= j_ed; ++j) {
            for (int i = i_st; i <= i_ed; ++i) {
                
                for (int l = 1; l <= ng; ++l) {
                    int idx_g[3] = {i, j, k};
                    int idx_i[3] = {i, j, k};
                    if (reg.s_lr == 1) { idx_g[dir]+=l; idx_i[dir]-=l; }
                    else               { idx_g[dir]-=l; idx_i[dir]+=l; }

                    double u_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 1);
                    double v_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 2);
                    double w_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 3);
                    double p_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 4);
                    double t_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 5);

                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = -u_i;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = -v_i;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = -w_i;

                    double p_g = p_i;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = p_g;

                    double t_g = isothermal ? (2.0 * twall_dim - t_i) : t_i;
                    if (t_g < 1e-6) t_g = t_i;
                    if (bf.prim.dims()[3] > 5) bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = t_g;

                    double rho_g = moocp * p_g / t_g;

                    // [TGH] 粘性壁面: 最外层密度取反
                    if (l == ng) bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = -rho_g;
                    else         bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = rho_g;
                }

                int idx_0[3] = {i,j,k};
                bf.prim(idx_0[0],idx_0[1],idx_0[2], 1) = 0.0;
                bf.prim(idx_0[0],idx_0[1],idx_0[2], 2) = 0.0;
                bf.prim(idx_0[0],idx_0[1],idx_0[2], 3) = 0.0;
                
                apply_dif_average(bf, i, j, k, dir, reg.s_lr, 6); 
            }
        }
    }
}

// ===========================================================================
// 4. 远场 (Farfield / boundary_4)
// 密度在最外层取反
// ===========================================================================
static void apply_farfield_bc(const BCRegion& reg, 
                              orion::preprocess::BlockField& bf,
                              const orion::core::Params& params,
                              int ng)
{
    int i_st = reg.s_st[0] - 1 + ng;
    int i_ed = reg.s_ed[0] - 1 + ng;
    int j_st = reg.s_st[1] - 1 + ng;
    int j_ed = reg.s_ed[1] - 1 + ng;
    int k_st = reg.s_st[2] - 1 + ng;
    int k_ed = reg.s_ed[2] - 1 + ng;

    int dir = reg.s_nd - 1;

    const auto& F = params.inflow;
    double gama = F.gama;
    double gm1  = gama - 1.0;
    
    double rho_inf = F.roo;
    double u_inf   = F.uoo;
    double v_inf   = F.voo;
    double w_inf   = F.woo;
    double p_inf   = F.poo;
    double c_inf = std::sqrt(gama * p_inf / rho_inf);

    for (int k = k_st; k <= k_ed; ++k) {
        for (int j = j_st; j <= j_ed; ++j) {
            for (int i = i_st; i <= i_ed; ++i) {
                
                double nx, ny, nz;
                get_normal(bf, i, j, k, reg.s_nd, nx, ny, nz);
                if (reg.s_lr == -1) { nx=-nx; ny=-ny; nz=-nz; } 

                int idx_i[3] = {i, j, k};
                if (reg.s_lr == 1) idx_i[dir] -= 1;
                else               idx_i[dir] += 1;

                double rho_in = bf.prim(idx_i[0], idx_i[1], idx_i[2], 0);
                double u_in   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 1);
                double v_in   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 2);
                double w_in   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 3);
                double p_in   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 4);

                double c_in = std::sqrt(gama * p_in / rho_in);
                double s_in = p_in / std::pow(rho_in, gama);
                double s_inf = p_inf / std::pow(rho_inf, gama);

                double vn_in  = u_in*nx  + v_in*ny  + w_in*nz;
                double vn_inf = u_inf*nx + v_inf*ny + w_inf*nz;
                double vel_mag = std::sqrt(u_in*u_in + v_in*v_in + w_in*w_in);
                double mach = vel_mag / c_in;

                double rho_b, u_b, v_b, w_b, p_b;

                if (mach < 1.0) { // Subsonic
                    double R_plus  = vn_in  + 2.0 * c_in  / gm1;
                    double R_minus = vn_inf - 2.0 * c_inf / gm1;
                    double vn_b = 0.5 * (R_plus + R_minus);
                    double c_b  = 0.25 * gm1 * (R_plus - R_minus);
                    
                    double s_b, u_ref, v_ref, w_ref, vn_ref;
                    if (vn_b > 0.0) { // Outflow
                        s_b = s_in;
                        u_ref = u_in; v_ref = v_in; w_ref = w_in; vn_ref = vn_in;
                    } else { // Inflow
                        s_b = s_inf;
                        u_ref = u_inf; v_ref = v_inf; w_ref = w_inf; vn_ref = vn_inf;
                    }
                    rho_b = std::pow(c_b*c_b / (gama * s_b), 1.0/gm1);
                    p_b   = s_b * std::pow(rho_b, gama);
                    u_b = u_ref + (vn_b - vn_ref) * nx;
                    v_b = v_ref + (vn_b - vn_ref) * ny;
                    w_b = w_ref + (vn_b - vn_ref) * nz;

                } else { // Supersonic
                    if (vn_in > 0.0) { // Outflow
                        rho_b = rho_in; u_b = u_in; v_b = v_in; w_b = w_in; p_b = p_in;
                    } else { // Inflow
                        rho_b = rho_inf; u_b = u_inf; v_b = v_inf; w_b = w_inf; p_b = p_inf;
                    }
                }
                
                double t_b = (mach*mach*gama) * p_b / rho_b; 

                for (int l = 1; l <= ng; ++l) {
                    int idx_g[3] = {i, j, k};
                    if (reg.s_lr == 1) idx_g[dir] += l;
                    else               idx_g[dir] -= l;

                    // [TGH] 远场: 最外层密度取反 (参见 boundary_4 代码)
                    if (l == ng) bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = -rho_b;
                    else         bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = rho_b;
                    
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = u_b;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = v_b;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = w_b;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = p_b;
                    if (bf.prim.dims()[3] > 5) bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = t_b;
                }
            }
        }
    }
}

// ===========================================================================
// 5. 强制无穷远边界 (boundary_5)
// ===========================================================================
static void apply_freestream_bc(const BCRegion& reg, 
                                orion::preprocess::BlockField& bf,
                                const orion::core::Params& params,
                                int ng)
{
    int i_st = reg.s_st[0] - 1 + ng;
    int i_ed = reg.s_ed[0] - 1 + ng;
    int j_st = reg.s_st[1] - 1 + ng;
    int j_ed = reg.s_ed[1] - 1 + ng;
    int k_st = reg.s_st[2] - 1 + ng;
    int k_ed = reg.s_ed[2] - 1 + ng;

    int dir = reg.s_nd - 1;

    const auto& F = params.inflow;
    double rho_inf = F.roo;
    double u_inf   = F.uoo;
    double v_inf   = F.voo;
    double w_inf   = F.woo;
    double p_inf   = F.poo;
    double t_inf   = F.too; 

    // boundary_5: 从 boundary point (Layer 0) 到 Ghost End
    int start_layer = (params.control.method == 1) ? 0 : 1;

    for (int k = k_st; k <= k_ed; ++k) {
        for (int j = j_st; j <= j_ed; ++j) {
            for (int i = i_st; i <= i_ed; ++i) {
                
                for (int l = start_layer; l <= ng; ++l) {
                    int idx_g[3] = {i, j, k};
                    if (reg.s_lr == 1) idx_g[dir] += l;
                    else               idx_g[dir] -= l;

                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = rho_inf;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = u_inf;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = v_inf;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = w_inf;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = p_inf;
                    if (bf.prim.dims()[3] > 5) bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = t_inf;
                }
            }
        }
    }
}

// ===========================================================================
// 6. 零梯度外推边界 (boundary_6)
// ===========================================================================
static void apply_extrapolation_bc(const BCRegion& reg, 
                                   orion::preprocess::BlockField& bf,
                                   const orion::core::Params& params,
                                   int ng)
{
    int i_st = reg.s_st[0] - 1 + ng;
    int i_ed = reg.s_ed[0] - 1 + ng;
    int j_st = reg.s_st[1] - 1 + ng;
    int j_ed = reg.s_ed[1] - 1 + ng;
    int k_st = reg.s_st[2] - 1 + ng;
    int k_ed = reg.s_ed[2] - 1 + ng;

    int dir = reg.s_nd - 1;

    for (int k = k_st; k <= k_ed; ++k) {
        for (int j = j_st; j <= j_ed; ++j) {
            for (int i = i_st; i <= i_ed; ++i) {
                
                int idx_src[3] = {i, j, k};
                if (reg.s_lr == 1) idx_src[dir] -= 1; 
                else               idx_src[dir] += 1;

                double rho = bf.prim(idx_src[0], idx_src[1], idx_src[2], 0);
                double u   = bf.prim(idx_src[0], idx_src[1], idx_src[2], 1);
                double v   = bf.prim(idx_src[0], idx_src[1], idx_src[2], 2);
                double w   = bf.prim(idx_src[0], idx_src[1], idx_src[2], 3);
                double p   = bf.prim(idx_src[0], idx_src[1], idx_src[2], 4);
                double t   = (bf.prim.dims()[3] > 5) ? bf.prim(idx_src[0], idx_src[1], idx_src[2], 5) : 0.0;

                int start_layer = (params.control.method == 1) ? 0 : 1;

                for (int l = start_layer; l <= ng; ++l) {
                    int idx_g[3] = {i, j, k};
                    if (reg.s_lr == 1) idx_g[dir] += l;
                    else               idx_g[dir] -= l;

                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = rho;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = u;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = v;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = w;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = p;
                    if (bf.prim.dims()[3] > 5) bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = t;
                }
            }
        }
    }
}

// ===========================================================================
// 主入口函数
// ===========================================================================
void apply_physical_bc(const BCData& bc, 
                       orion::preprocess::FlowFieldSet& fs,
                       const orion::core::Params& params)
{
    const int ng = fs.ng;

    for (int nb_idx : fs.local_block_ids) {
        int nb = nb_idx; 
        auto& bf = fs.blocks[nb];
        const auto& bcb = bc.block_bc[nb];

        for (const auto& reg : bcb.regions) {
            
            if (reg.bctype < 0) continue; 

            switch (reg.bctype) {
                case 2: // 通用壁面 (Viscid/Inviscid)
                    if (params.flowtype.nvis == 1) apply_viscid_wall(reg, bf, params, ng);
                    else                           apply_inviscid_wall(reg, bf, params, ng);
                    break;
                case 3: // 对称面
                    apply_symmetry_bc(reg, bf, params, ng);
                    break;
                case 4: // 远场
                    apply_farfield_bc(reg, bf, params, ng);
                    break;
                case 5: // 强制无穷远
                    apply_freestream_bc(reg, bf, params, ng);
                    break;
                case 6: // 零梯度外推
                    apply_extrapolation_bc(reg, bf, params, ng);
                    break;
                default:
                    break;
            }
        }
    }
}

} // namespace orion::bc