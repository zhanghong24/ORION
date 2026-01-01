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
    // s_nd 是 1-based (1,2,3)，转为 0-based offset
    // 注意：bf.metrics 访问时也需要加上 ng 偏移！
    // 假设 metrics 和 prim 一样，物理点 (0,0,0) 对应内存 (ng,ng,ng)
    // 需要外部调用者保证传入的 i,j,k 是 0-based 物理坐标
    
    // 获取当前 FlowField 的 ng (需要从 bf 获取，或者假设调用者处理好了)
    // 这里的 i,j,k 是物理坐标。bf.metrics(...) 通常设计为接受物理坐标吗？
    // 查看 MetricComputer.cpp -> metrics 是 allocated (ni+2ng)...
    // 我们需要统一标准：这里的 helper 函数接受 "物理坐标 i,j,k"，内部加 ng。
    
    // 为了安全，我们需要知道 ng。但是 BlockField 结构里好像没有 ng 成员？
    // 通常 ng 存储在 Params 或 FlowFieldSet 中。
    // 既然外部调用者 (apply_...) 知道 ng，我们让 helper 接收实际内存索引 idx_i, idx_j, idx_k 更好。
    // 但为了保持接口简单，我们假设传入的 i,j,k 已经是 "内存索引" (即 phys + ng)。
    
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
                               int idx_i_st, int idx_i_ed,
                               int idx_j_st, int idx_j_ed,
                               int idx_k_st, int idx_k_ed,
                               int s_nd,
                               double& nx_avg, double& ny_avg, double& nz_avg)
{
    int i_mid = (idx_i_st + idx_i_ed) / 2;
    int j_mid = (idx_j_st + idx_j_ed) / 2;
    int k_mid = (idx_k_st + idx_k_ed) / 2;

    double nx, ny, nz;
    get_normal(bf, i_mid, j_mid, k_mid, s_nd, nx, ny, nz);
    
    nx_avg = nx;
    ny_avg = ny;
    nz_avg = nz;
}

// ===========================================================================
// 辅助函数：dif_average (method=1 专用)
// ===========================================================================
static void apply_dif_average(orion::preprocess::BlockField& bf,
                              int idx_i, int idx_j, int idx_k, // 内存索引
                              int dir, int s_lr, int nprim)
{
    int idx_0[3] = {idx_i, idx_j, idx_k}; // Face Point
    int idx_g[3] = {idx_i, idx_j, idx_k}; // Ghost Layer 1
    int idx_i_inner[3] = {idx_i, idx_j, idx_k}; // Inner Layer 1

    if (s_lr == 1) { // Max Face
        idx_g[dir] += 1;
        idx_i_inner[dir] -= 1;
    } else { // Min Face
        idx_g[dir] -= 1;
        idx_i_inner[dir] += 1;
    }

    for (int m = 0; m < nprim; ++m) {
        double val_g = bf.prim(idx_g[0], idx_g[1], idx_g[2], m);
        double val_i = bf.prim(idx_i_inner[0], idx_i_inner[1], idx_i_inner[2], m);
        bf.prim(idx_0[0], idx_0[1], idx_0[2], m) = 0.5 * (val_g + val_i);
    }
}

// ===========================================================================
// 1. 对称边界 (Symmetry / boundary_3)
// ===========================================================================
static void apply_symmetry_bc(const BCRegion& reg, 
                              orion::preprocess::BlockField& bf,
                              const orion::core::Params& params,
                              int ng)
{
    // s_st/ed 是 1-based。转换为 0-based 物理坐标，再加 ng 转换为内存坐标。
    int i_st = reg.s_st[0] - 1 + ng;
    int i_ed = reg.s_ed[0] - 1 + ng;
    int j_st = reg.s_st[1] - 1 + ng;
    int j_ed = reg.s_ed[1] - 1 + ng;
    int k_st = reg.s_st[2] - 1 + ng;
    int k_ed = reg.s_ed[2] - 1 + ng;

    int dir = reg.s_nd - 1; // 0-based direction (0=i, 1=j, 2=k)

    // 判断是否为 2D 算例 (k方向维度包含ghost, 物理维度 = dim - 2*ng)
    int phys_dim_k = bf.prim.dims()[2] - 2 * ng;
    bool is_2d_symmetry = (dir != 2 && phys_dim_k <= 2);

    if (is_2d_symmetry) {
        for (int k = k_st; k <= k_ed; ++k) {
            for (int j = j_st; j <= j_ed; ++j) {
                for (int i = i_st; i <= i_ed; ++i) {
                    
                    for (int l = 0; l < ng; ++l) { // 0-based loop
                        int dist = l + 1; // 1-based distance
                        int idx_g[3] = {i, j, k};
                        int idx_i[3] = {i, j, k};
                        
                        if (reg.s_lr == 1) { idx_g[dir]+=dist; idx_i[dir]-=dist; }
                        else               { idx_g[dir]-=dist; idx_i[dir]+=dist; }

                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 0);
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 1);
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 2);
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 4);
                        if (bf.prim.dims()[3] > 5) 
                             bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 5);

                        // W 反向 (展向)
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = -bf.prim(idx_i[0], idx_i[1], idx_i[2], 3); 
                    }
                    
                    if (params.control.method == 1) 
                        apply_dif_average(bf, i, j, k, dir, reg.s_lr, bf.prim.dims()[3]);
                }
            }
        }
    } else {
        double nx, ny, nz;
        get_average_normal(bf, i_st, i_ed, j_st, j_ed, k_st, k_ed, reg.s_nd, nx, ny, nz);

        for (int k = k_st; k <= k_ed; ++k) {
            for (int j = j_st; j <= j_ed; ++j) {
                for (int i = i_st; i <= i_ed; ++i) {
                    
                    for (int l = 0; l < ng; ++l) {
                        int dist = l + 1;
                        int idx_g[3] = {i,j,k};
                        int idx_i[3] = {i,j,k};
                        if (reg.s_lr == 1) { idx_g[dir]+=dist; idx_i[dir]-=dist; }
                        else               { idx_g[dir]-=dist; idx_i[dir]+=dist; }

                        double rho = bf.prim(idx_i[0], idx_i[1], idx_i[2], 0);
                        double u   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 1);
                        double v   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 2);
                        double w   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 3);
                        double p   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 4);

                        double v_dot_n = u*nx + v*ny + w*nz;

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
// 100% 复刻 Fortran: 
// - Layer 1, 2: 正向
// - Layer 3: 密度取反
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
                
                // 注意：get_normal 内部接收的是内存坐标 (i,j,k)
                double nx, ny, nz;
                get_normal(bf, i, j, k, reg.s_nd, nx, ny, nz);

                for (int l = 0; l < ng; ++l) { 
                    int dist = l + 1; // 1-based 距离
                    int idx_g[3] = {i, j, k};
                    int idx_i[3] = {i, j, k};

                    if (reg.s_lr == 1) { idx_g[dir]+=dist; idx_i[dir]-=dist; }
                    else               { idx_g[dir]-=dist; idx_i[dir]+=dist; }

                    double rho_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 0);
                    double u_i   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 1);
                    double v_i   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 2);
                    double w_i   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 3);
                    double p_i   = bf.prim(idx_i[0], idx_i[1], idx_i[2], 4);
                    double t_i   = (bf.prim.dims()[3] > 5) ? bf.prim(idx_i[0], idx_i[1], idx_i[2], 5) : 0.0;

                    double v_dot_n = u_i*nx + v_i*ny + w_i*nz;
                    
                    // [100% 复刻] 第3层密度取反 (参见 BC.f90 line 95)
                    // 如果 ng < 3，这个分支不会触发，程序安全
                    if (dist == 3) bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = -rho_i;
                    else           bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = rho_i;
                    
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = u_i - 2.0 * v_dot_n * nx;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = v_i - 2.0 * v_dot_n * ny;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = w_i - 2.0 * v_dot_n * nz;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = p_i;
                    
                    if (bf.prim.dims()[3] > 5) bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = t_i;
                }

                if (params.control.method == 1) 
                    apply_dif_average(bf, i, j, k, dir, reg.s_lr, bf.prim.dims()[3]);
            }
        }
    }
}

// ===========================================================================
// 3. 粘性壁面 (Viscous Wall)
// 100% 复刻 Fortran: 第3层密度取反 + 2阶插值
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
    const double pmin_limit = 1.0e-6; 
    
    bool isothermal = (twall > 1.0e-6);
    double twall_dim = (isothermal && std::abs(tref)>1e-16) ? twall/tref : twall;

    for (int k = k_st; k <= k_ed; ++k) {
        for (int j = j_st; j <= j_ed; ++j) {
            for (int i = i_st; i <= i_ed; ++i) {
                
                // 1. 计算壁面 (Layer 0) 物理值 - 2阶插值
                int idx_1[3] = {i, j, k}; // Inner 1 (dist=1)
                int idx_2[3] = {i, j, k}; // Inner 2 (dist=2)
                
                if (reg.s_lr == 1) { 
                    idx_1[dir] -= 1; idx_2[dir] -= 2; 
                } else { 
                    idx_1[dir] += 1; idx_2[dir] += 2; 
                }

                double p1 = bf.prim(idx_1[0], idx_1[1], idx_1[2], 4);
                double p2 = bf.prim(idx_2[0], idx_2[1], idx_2[2], 4);
                double p_wall = (4.0 * p1 - p2) / 3.0;
                if (p_wall < pmin_limit) p_wall = p1; 

                double t_wall;
                if (isothermal) {
                    t_wall = twall_dim;
                } else {
                    double t1 = bf.prim(idx_1[0], idx_1[1], idx_1[2], 5);
                    double t2 = bf.prim(idx_2[0], idx_2[1], idx_2[2], 5);
                    t_wall = (4.0 * t1 - t2) / 3.0;
                    if (t_wall < 1e-6) t_wall = t1;
                }

                double rho_wall = moocp * p_wall / t_wall;

                // 2. 填充 Ghost Layers (Layer 1..ng)
                for (int l = 0; l < ng; ++l) {
                    int dist = l + 1;
                    int idx_g[3] = {i, j, k};
                    int idx_i[3] = {i, j, k};
                    if (reg.s_lr == 1) { idx_g[dir]+=dist; idx_i[dir]-=dist; }
                    else               { idx_g[dir]-=dist; idx_i[dir]+=dist; }

                    double u_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 1);
                    double v_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 2);
                    double w_i = bf.prim(idx_i[0], idx_i[1], idx_i[2], 3);

                    // No-slip condition
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 1) = -u_i;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 2) = -v_i;
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 3) = -w_i;

                    // P: 零梯度近似 (Fortran line 68)
                    bf.prim(idx_g[0], idx_g[1], idx_g[2], 4) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 4);
                    
                    if (isothermal) {
                        double t_inner = bf.prim(idx_i[0], idx_i[1], idx_i[2], 5);
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = 2.0*t_wall - t_inner;
                    } else {
                        bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) = bf.prim(idx_i[0], idx_i[1], idx_i[2], 5);
                    }

                    double pg = bf.prim(idx_g[0], idx_g[1], idx_g[2], 4);
                    double tg = (bf.prim.dims()[3] > 5) ? bf.prim(idx_g[0], idx_g[1], idx_g[2], 5) : 1.0;
                    if (tg < 1e-6) tg = 1.0;
                    
                    double rho_calc = moocp * pg / tg;

                    // [100% 复刻] 第3层密度取反 (BC.f90 line 74)
                    if (dist == 3) bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = -rho_calc;
                    else           bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = rho_calc;
                }

                // 3. 强制更新 Boundary Point (Layer 0)
                // 这一点很重要，因为粘性通量计算需要壁面上的准确值
                int idx_0[3] = {i,j,k};
                bf.prim(idx_0[0], idx_0[1], idx_0[2], 0) = rho_wall;
                bf.prim(idx_0[0], idx_0[1], idx_0[2], 1) = 0.0;
                bf.prim(idx_0[0], idx_0[1], idx_0[2], 2) = 0.0;
                bf.prim(idx_0[0], idx_0[1], idx_0[2], 3) = 0.0;
                bf.prim(idx_0[0], idx_0[1], idx_0[2], 4) = p_wall;
                if (bf.prim.dims()[3] > 5) 
                    bf.prim(idx_0[0], idx_0[1], idx_0[2], 5) = t_wall;
                
                if (params.control.method == 1) 
                    apply_dif_average(bf, i, j, k, dir, reg.s_lr, 6); 
            }
        }
    }
}

// ===========================================================================
// 4. 远场 (Farfield / boundary_4)
// 100% 复刻 Fortran: 第3层密度取反
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

                for (int l = 0; l < ng; ++l) {
                    int dist = l + 1;
                    int idx_g[3] = {i, j, k};
                    if (reg.s_lr == 1) idx_g[dir] += dist;
                    else               idx_g[dir] -= dist;

                    // [100% 复刻] 第3层密度取反 (BC.f90 line 47)
                    if (dist == 3) bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = -rho_b;
                    else           bf.prim(idx_g[0], idx_g[1], idx_g[2], 0) = rho_b;
                    
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

    // boundary_5: Fortran 逻辑也覆盖了 Ghost Layers
    // 假设简单填满
    for (int k = k_st; k <= k_ed; ++k) {
        for (int j = j_st; j <= j_ed; ++j) {
            for (int i = i_st; i <= i_ed; ++i) {
                
                for (int l = 0; l < ng; ++l) {
                    int dist = l + 1;
                    int idx_g[3] = {i, j, k};
                    if (reg.s_lr == 1) idx_g[dir] += dist;
                    else               idx_g[dir] -= dist;

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

                for (int l = 0; l < ng; ++l) {
                    int dist = l + 1;
                    int idx_g[3] = {i, j, k};
                    if (reg.s_lr == 1) idx_g[dir] += dist;
                    else               idx_g[dir] -= dist;

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