#include "solver/FluxComputer.hpp"
#include "core/Runtime.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

namespace orion::solver {

// ===========================================================================
// [Helper 1] DUVWT_NODE_LINE_virtual (导数计算 - 节点)
// ===========================================================================
static void compute_node_deriv_line(int n, int ni, const std::vector<double>& q, 
                                    std::vector<double>& dq, int flag_left, int flag_right) 
{
    const double dd12 = 12.0, dd6 = 6.0;
    const double A1 = 8.0, B1 = -1.0;
    const double A2 = -3.0, B2 = -10.0, C2 = 18.0, D2 = -6.0, E2 = 1.0;

    if (ni <= 2) {
        for (int i = 1; i <= ni; ++i) {
            for (int m = 0; m < n; ++m) {
                dq[(i-1)*n+m] = 0.5 * (q[(i+3)*n+m] - q[(i+1)*n+m]);
            }
        }
        return;
    }

    int st = 1;
    int ed = ni;

    // 左边界偏心处理
    if (flag_left < 0) {
        st = 3;
        for (int m = 0; m < n; ++m) {
            double q1 = q[3*n+m], q2 = q[4*n+m], q3 = q[5*n+m], q4 = q[6*n+m], q5 = q[7*n+m];
            dq[0*n+m] = (-11.0*q1 + 18.0*q2 - 9.0*q3 + 2.0*q4) / dd6;
            dq[1*n+m] = (A2*q1 + B2*q2 + C2*q3 + D2*q4 + E2*q5) / dd12;
        }
    }

    // 右边界偏心处理
    if (flag_right < 0) {
        ed = ni - 2;
        for (int m = 0; m < n; ++m) {
            double qn   = q[(ni+2)*n+m];
            double qn_1 = q[(ni+1)*n+m];
            double qn_2 = q[(ni+0)*n+m];
            double qn_3 = q[(ni-1)*n+m];
            double qn_4 = q[(ni-2)*n+m];
            
            dq[(ni-1)*n+m] = -(-11.0*qn + 18.0*qn_1 - 9.0*qn_2 + 2.0*qn_3) / dd6;
            dq[(ni-2)*n+m] = -(A2*qn + B2*qn_1 + C2*qn_2 + D2*qn_3 + E2*qn_4) / dd12;
        }
    }

    // 内部 4阶中心差分
    for (int i = st; i <= ed; ++i) {
        for (int m = 0; m < n; ++m) {
            double q_p1 = q[(i+3)*n+m];
            double q_m1 = q[(i+1)*n+m];
            double q_p2 = q[(i+4)*n+m];
            double q_m2 = q[(i+0)*n+m];
            dq[(i-1)*n+m] = (A1*(q_p1 - q_m1) + B1*(q_p2 - q_m2)) / 12.0;
        }
    }
}

// ===========================================================================
// [Helper 2] DUVWT_half_line_virtual (导数计算 - 半点)
// ===========================================================================
static void compute_half_deriv_line(int n, int ni, const std::vector<double>& q, 
                                    std::vector<double>& dq, int flag_left, int flag_right)
{
    const double dd24 = 24.0;
    const double A1=27.0, B1=-1.0;
    const double A2=-22.0, B2=17.0, C2=9.0, D2=-5.0, E2=1.0;
    const double A3=-71.0, B3=141.0, C3=-93.0, D3=23.0;

    if (ni <= 2) {
        for (int i = 0; i <= ni; ++i) {
            for (int m = 0; m < n; ++m) {
                dq[i*n+m] = q[(i+3)*n+m] - q[(i+2)*n+m];
            }
        }
        return;
    }

    int st = 0;
    int ed = ni;

    if (flag_left < 0) {
        st = 2;
        for (int m = 0; m < n; ++m) {
            double q1 = q[3*n+m], q2 = q[4*n+m], q3 = q[5*n+m], q4 = q[6*n+m], q5 = q[7*n+m];
            dq[0*n+m] = (A3*q1 + B3*q2 + C3*q3 + D3*q4) / dd24;
            dq[1*n+m] = (A2*q1 + B2*q2 + C2*q3 + D2*q4 + E2*q5) / dd24;
        }
    }

    if (flag_right < 0) {
        ed = ni - 2;
        for (int m = 0; m < n; ++m) {
            double qn   = q[(ni+2)*n+m];
            double qn_1 = q[(ni+1)*n+m];
            double qn_2 = q[(ni+0)*n+m];
            double qn_3 = q[(ni-1)*n+m];
            double qn_4 = q[(ni-2)*n+m];
            
            dq[ni*n+m]     = -(A3*qn + B3*qn_1 + C3*qn_2 + D3*qn_3) / dd24;
            dq[(ni-1)*n+m] = -(A2*qn + B2*qn_1 + C2*qn_2 + D2*qn_3 + E2*qn_4) / dd24;
        }
    }

    for (int i = st; i <= ed; ++i) {
        for (int m = 0; m < n; ++m) {
            double q_i   = q[(i+2)*n+m];
            double q_ip1 = q[(i+3)*n+m];
            double q_ip2 = q[(i+4)*n+m];
            double q_im1 = q[(i+1)*n+m];
            dq[i*n+m] = (A1*(q_ip1 - q_i) + B1*(q_ip2 - q_im1)) / dd24;
        }
    }
}

// ===========================================================================
// [Helper 3] 插值函数 (节点 -> 半点)
// ===========================================================================
static void interp_line_to_half(int n, int ni, const std::vector<double>& q, 
                                std::vector<double>& val, int flag_left, int flag_right)
{
    const double dd16=16.0;
    const double A1=9.0, B1=-1.0;
    const double A2=5.0, B2=15.0, C2=-5.0, D2=1.0;
    const double A3=35.0, B3=-35.0, C3=21.0, D3=-5.0;

    if (ni <= 2) {
        for (int i = 0; i <= ni; ++i) {
            for (int m = 0; m < n; ++m) 
                val[i*n+m] = 0.5 * (q[(i+2)*n+m] + q[(i+3)*n+m]);
        }
        return;
    }

    int st = 0;
    int ed = ni;

    if (flag_left < 0) {
        st = 2;
        for (int m = 0; m < n; ++m) {
            double q1 = q[3*n+m], q2 = q[4*n+m], q3 = q[5*n+m], q4 = q[6*n+m];
            val[0*n+m] = (A3*q1 + B3*q2 + C3*q3 + D3*q4) / dd16;
            val[1*n+m] = (A2*q1 + B2*q2 + C2*q3 + D2*q4) / dd16;
        }
    }

    if (flag_right < 0) {
        ed = ni - 2;
        for (int m = 0; m < n; ++m) {
            double qn   = q[(ni+2)*n+m];
            double qn_1 = q[(ni+1)*n+m];
            double qn_2 = q[(ni+0)*n+m];
            double qn_3 = q[(ni-1)*n+m];
            val[ni*n+m]     = (A3*qn + B3*qn_1 + C3*qn_2 + D3*qn_3) / dd16;
            val[(ni-1)*n+m] = (A2*qn + B2*qn_1 + C2*qn_2 + D2*qn_3) / dd16;
        }
    }

    for (int i = st; i <= ed; ++i) {
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
// [Helper 4] 坐标变换 (不变)
// ===========================================================================
static void transform_derivs_to_phys(int ni, const std::vector<double>& duvwt, 
                                     const std::vector<double>& kxyz, 
                                     const std::vector<double>& vol,
                                     std::vector<double>& duvwtdxyz)
{
    for (int i = 0; i <= ni; ++i) {
        double vol_inv = 1.0 / vol[i];
        
        for (int m = 0; m < 3; ++m) { 
            double k1 = kxyz[i*9 + m];     // xi_xm
            double k2 = kxyz[i*9 + m + 3]; // eta_xm
            double k3 = kxyz[i*9 + m + 6]; // zeta_xm

            for (int v_idx = 0; v_idx < 4; ++v_idx) {
                double d_xi   = duvwt[i*12 + v_idx];     
                double d_eta  = duvwt[i*12 + v_idx + 4]; 
                double d_zeta = duvwt[i*12 + v_idx + 8]; 
                
                duvwtdxyz[i*12 + v_idx*3 + m] = (k1*d_xi + k2*d_eta + k3*d_zeta) * vol_inv;
            }
        }
    }
}

// ===========================================================================
// [Helper 5] 计算物理粘性通量 (不变)
// ===========================================================================
static void compute_viscous_flux_1d(int ni, const std::vector<double>& uvwt, 
                                    const std::vector<double>& duvwtdxyz,
                                    const std::vector<double>& vslt1, 
                                    const std::vector<double>& vslt2,
                                    const std::vector<double>& normal_vec,
                                    std::vector<double>& fv)
{
    const double CC = 2.0 / 3.0;
    
    for (int i = 0; i <= ni; ++i) {
        double vs = vslt1[i];
        double kcp = vslt2[i];
        double vscc = vs * CC;

        double dudx = duvwtdxyz[i*12 + 0], dudy = duvwtdxyz[i*12 + 1], dudz = duvwtdxyz[i*12 + 2];
        double dvdx = duvwtdxyz[i*12 + 3], dvdy = duvwtdxyz[i*12 + 4], dvdz = duvwtdxyz[i*12 + 5];
        double dwdx = duvwtdxyz[i*12 + 6], dwdy = duvwtdxyz[i*12 + 7], dwdz = duvwtdxyz[i*12 + 8];

        double txx = vscc * (2.0*dudx - dvdy - dwdz);
        double tyy = vscc * (2.0*dvdy - dwdz - dudx);
        double tzz = vscc * (2.0*dwdz - dudx - dvdy);
        
        double txy = vs * (dudy + dvdx);
        double txz = vs * (dudz + dwdx);
        double tyz = vs * (dvdz + dwdy);

        double nx = normal_vec[i*3 + 0];
        double ny = normal_vec[i*3 + 1];
        double nz = normal_vec[i*3 + 2];

        fv[i*5 + 0] = 0.0;
        fv[i*5 + 1] = txx*nx + txy*ny + txz*nz;
        fv[i*5 + 2] = txy*nx + tyy*ny + tyz*nz;
        fv[i*5 + 3] = txz*nx + tyz*ny + tzz*nz;

        double u = uvwt[i*4 + 0], v = uvwt[i*4 + 1], w = uvwt[i*4 + 2];
        double work = u*fv[i*5 + 1] + v*fv[i*5 + 2] + w*fv[i*5 + 3];
        
        double heat = kcp * (duvwtdxyz[i*12 + 9]*nx + duvwtdxyz[i*12 + 10]*ny + duvwtdxyz[i*12 + 11]*nz);

        fv[i*5 + 4] = work + heat;
    }
}

// ===========================================================================
// [Helper 6] FLUX_DXYZ (复刻 Fortran 高阶差分)
// ===========================================================================
static void compute_flux_diff_1d(int nl, int ni, const std::vector<double>& fv, 
                                 std::vector<double>& dfv)
{
    // 如果网格太少，退化为 2 阶
    if (ni <= 2) {
        for (int i = 0; i < ni; ++i) {
            for (int m = 0; m < nl; ++m) {
                dfv[i*nl + m] = fv[(i+1)*nl + m] - fv[i*nl + m];
            }
        }
        return;
    }

    // 内部: 6阶差分
    for (int i = 2; i < ni - 2; ++i) { // 0-based: 2 to ni-3
        for (int m = 0; m < nl; ++m) {
            double term1 = 2250.0 * (fv[(i+1)*nl+m] - fv[i*nl+m]);
            double term2 = -125.0 * (fv[(i+2)*nl+m] - fv[(i-1)*nl+m]);
            double term3 =    9.0 * (fv[(i+3)*nl+m] - fv[(i-2)*nl+m]);
            dfv[i*nl+m] = (term1 + term2 + term3) / 1920.0;
        }
    }

    // 边界处理 (0-based indices)
    for (int m = 0; m < nl; ++m) {
        // i = 1 (Fortran 2)
        // Fortran: ( f(0) - 27*f(1) + 27*f(2) - f(3) ) / 24
        // C++: fv has 0-based indexing matching Fortran relative order
        dfv[1*nl+m] = (fv[0*nl+m] - 27.0*fv[1*nl+m] + 27.0*fv[2*nl+m] - fv[3*nl+m]) / 24.0;
        
        // i = ni-2 (Fortran ni-1)
        dfv[(ni-2)*nl+m] = -(fv[ni*nl+m] - 27.0*fv[(ni-1)*nl+m] + 27.0*fv[(ni-2)*nl+m] - fv[(ni-3)*nl+m]) / 24.0;
        
        // i = 0 (Fortran 1) -> 3rd Order
        dfv[0*nl+m] = (-23.0*fv[0*nl+m] + 21.0*fv[1*nl+m] + 3.0*fv[2*nl+m] - fv[3*nl+m]) / 24.0;

        // i = ni-1 (Fortran ni) -> 3rd Order
        dfv[(ni-1)*nl+m] = -(-23.0*fv[ni*nl+m] + 21.0*fv[(ni-1)*nl+m] + 3.0*fv[(ni-2)*nl+m] - fv[(ni-3)*nl+m]) / 24.0;
    }
}

// ===========================================================================
// 类成员：计算导数 (Cell)
// ===========================================================================
void FluxComputer::compute_derivatives_cell(const orion::preprocess::BlockField& bf, 
                                            orion::OrionArray<double>& duvwt, 
                                            int ng) 
{
    const auto& dims = bf.prim.dims();
    int nx = dims[0], ny = dims[1], nz = dims[2];
    int idim = nx - 2*ng, jdim = ny - 2*ng, kdim = nz - 2*ng;

    // --- I Direction ---
    std::vector<double> q_line((idim + 6) * 4); 
    std::vector<double> dq_line(idim * 4);

    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            // 使用 ng-3 检查最外层 Ghost
            int f_l = (bf.prim(ng-3, j, k, 0) < 0) ? -1 : 0;
            int f_r = (bf.prim(ng+idim+2, j, k, 0) < 0) ? -1 : 0;

            for (int i = -2; i < idim + 3; ++i) {
                int idx = std::max(0, std::min(nx-1, ng + i));
                for(int m=0; m<4; ++m) {
                    double val = (m<3) ? bf.prim(idx,j,k,m+1) : bf.prim(idx,j,k,5);
                    q_line[(i+3)*4 + m] = val;
                }
            }
            compute_node_deriv_line(4, idim, q_line, dq_line, f_l, f_r);
            for(int i=0; i<idim; ++i) {
                for(int m=0; m<4; ++m) duvwt(ng+i,j,k, m) = dq_line[i*4+m];
            }
        }
    }

    // --- J Direction ---
    std::vector<double> q_line_j((jdim + 6) * 4);
    std::vector<double> dq_line_j(jdim * 4);
    for (int k = ng; k < nz - ng; ++k) {
        for (int i = ng; i < nx - ng; ++i) {
            int f_l = (bf.prim(i, ng-3, k, 0) < 0) ? -1 : 0;
            int f_r = (bf.prim(i, ng+jdim+2, k, 0) < 0) ? -1 : 0;
            for (int j = -2; j < jdim + 3; ++j) {
                int idx = std::max(0, std::min(ny-1, ng + j));
                for(int m=0; m<4; ++m) {
                    // [修正] T 在 prim(..., 5)
                    double val = (m<3) ? bf.prim(i,idx,k,m+1) : bf.prim(i,idx,k,5);
                    q_line_j[(j+3)*4 + m] = val;
                }
            }
            compute_node_deriv_line(4, jdim, q_line_j, dq_line_j, f_l, f_r);
            for(int j=0; j<jdim; ++j) {
                for(int m=0; m<4; ++m) duvwt(i,ng+j,k, m+4) = dq_line_j[j*4+m];
            }
        }
    }

    // --- K Direction ---
    std::vector<double> q_line_k((kdim + 6) * 4);
    std::vector<double> dq_line_k(kdim * 4);
    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            int f_l = (bf.prim(i, j, ng-3, 0) < 0) ? -1 : 0;
            int f_r = (bf.prim(i, j, ng+kdim+2, 0) < 0) ? -1 : 0;
            for (int k = -2; k < kdim + 3; ++k) {
                int idx = std::max(0, std::min(nz-1, ng + k));
                for(int m=0; m<4; ++m) {
                    // [修正] T 在 prim(..., 5)
                    double val = (m<3) ? bf.prim(i,j,idx,m+1) : bf.prim(i,j,idx,5);
                    q_line_k[(k+3)*4 + m] = val;
                }
            }
            compute_node_deriv_line(4, kdim, q_line_k, dq_line_k, f_l, f_r);
            for(int k=0; k<kdim; ++k) {
                for(int m=0; m<4; ++m) duvwt(i,j,ng+k, m+8) = dq_line_k[k*4+m];
            }
        }
    }
}

// ===========================================================================
// 类成员：计算导数 (Face)
// ===========================================================================
void FluxComputer::compute_derivatives_face(const orion::preprocess::BlockField& bf, 
                                            orion::OrionArray<double>& duvwt_mid, 
                                            int ng) 
{
    const auto& dims = bf.prim.dims();
    int nx = dims[0], ny = dims[1], nz = dims[2];
    int idim = nx - 2*ng, jdim = ny - 2*ng, kdim = nz - 2*ng;

    // --- I Direction ---
    std::vector<double> q_line((idim + 6) * 4); 
    std::vector<double> dq_line((idim + 1) * 4); 

    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            int f_l = (bf.prim(ng-3, j, k, 0) < 0) ? -1 : 0;
            int f_r = (bf.prim(ng+idim+2, j, k, 0) < 0) ? -1 : 0;
            for (int i = -2; i < idim + 3; ++i) {
                int idx = std::max(0, std::min(nx-1, ng + i));
                for(int m=0; m<4; ++m) {
                    // [修正] T 在 prim(..., 5)
                    double val = (m<3) ? bf.prim(idx,j,k,m+1) : bf.prim(idx,j,k,5);
                    q_line[(i+3)*4 + m] = val;
                }
            }
            compute_half_deriv_line(4, idim, q_line, dq_line, f_l, f_r);
            for(int i=0; i<=idim; ++i) {
                int store_idx = ng + i - 1; 
                if(store_idx >= 0 && store_idx < nx)
                    for(int m=0; m<4; ++m) duvwt_mid(store_idx, j, k, m) = dq_line[i*4+m];
            }
        }
    }
    
    // --- J Direction ---
    std::vector<double> q_line_j((jdim + 6) * 4); 
    std::vector<double> dq_line_j((jdim + 1) * 4); 
    for (int k = ng; k < nz - ng; ++k) {
        for (int i = ng; i < nx - ng; ++i) {
            int f_l = (bf.prim(i, ng-3, k, 0) < 0) ? -1 : 0;
            int f_r = (bf.prim(i, ng+jdim+2, k, 0) < 0) ? -1 : 0;
            for (int j = -2; j < jdim + 3; ++j) {
                int idx = std::max(0, std::min(ny-1, ng + j));
                for(int m=0; m<4; ++m) {
                    // [修正] T 在 prim(..., 5)
                    double val = (m<3) ? bf.prim(i,idx,k,m+1) : bf.prim(i,idx,k,5);
                    q_line_j[(j+3)*4 + m] = val;
                }
            }
            compute_half_deriv_line(4, jdim, q_line_j, dq_line_j, f_l, f_r);
            for(int j=0; j<=jdim; ++j) {
                int store_idx = ng + j - 1;
                if(store_idx >= 0 && store_idx < ny)
                    for(int m=0; m<4; ++m) duvwt_mid(i, store_idx, k, m+4) = dq_line_j[j*4+m];
            }
        }
    }

    // --- K Direction ---
    std::vector<double> q_line_k((kdim + 6) * 4); 
    std::vector<double> dq_line_k((kdim + 1) * 4); 
    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            int f_l = (bf.prim(i, j, ng-3, 0) < 0) ? -1 : 0;
            int f_r = (bf.prim(i, j, ng+kdim+2, 0) < 0) ? -1 : 0;
            for (int k = -2; k < kdim + 3; ++k) {
                int idx = std::max(0, std::min(nz-1, ng + k));
                for(int m=0; m<4; ++m) {
                    // [修正] T 在 prim(..., 5)
                    double val = (m<3) ? bf.prim(i,j,idx,m+1) : bf.prim(i,j,idx,5);
                    q_line_k[(k+3)*4 + m] = val;
                }
            }
            compute_half_deriv_line(4, kdim, q_line_k, dq_line_k, f_l, f_r);
            for(int k=0; k<=kdim; ++k) {
                int store_idx = ng + k - 1;
                if(store_idx >= 0 && store_idx < nz)
                    for(int m=0; m<4; ++m) duvwt_mid(i, j, store_idx, m+8) = dq_line_k[k*4+m];
            }
        }
    }
}

// ===========================================================================
// 核心实现：计算粘性通量块
// ===========================================================================
void FluxComputer::compute_viscous_flux_block(orion::preprocess::BlockField& bf,
                                              const orion::core::Params& params,
                                              int ng)
{
    const auto& dims = bf.prim.dims();
    int nx = dims[0], ny = dims[1], nz = dims[2];
    int idim = nx - 2*ng, jdim = ny - 2*ng, kdim = nz - 2*ng;

    orion::OrionArray<double> duvwt(nx, ny, nz, 12);
    orion::OrionArray<double> duvwt_mid(nx, ny, nz, 12); 

    double reynolds = params.inflow.reynolds;
    double re_inv = (std::abs(reynolds) > 1.0e-30) ? 1.0 / reynolds : 0.0;
    double gama = params.inflow.gama;
    double moo = params.inflow.moo;
    double prl = params.inflow.prl;
    double cp = 1.0 / ((gama - 1.0) * moo * moo);
    double cp_prl = cp / prl;

    compute_derivatives_cell(bf, duvwt, ng);
    compute_derivatives_face(bf, duvwt_mid, ng);

    // --- I Direction Loop ---
    int n_face = idim + 1; 
    std::vector<double> uvwt_half(4 * n_face);
    std::vector<double> duvwt_half(12 * n_face);
    std::vector<double> kxyz_half(9 * n_face);
    std::vector<double> vol_half(n_face);
    std::vector<double> vslt1_half(n_face);
    std::vector<double> vslt2_half(n_face);
    std::vector<double> fv(5 * n_face); 
    std::vector<double> dfv(5 * idim);
    std::vector<double> normal_vec(3 * n_face); 
    
    std::vector<double> q_line((idim+6)*9); 

    for (int k = ng; k < nz - ng; ++k) {
        for (int j = ng; j < ny - ng; ++j) {
            int f_l = (bf.prim(ng-3, j, k, 0) < 0) ? -1 : 0;
            int f_r = (bf.prim(ng+idim+2, j, k, 0) < 0) ? -1 : 0;

            // UVWT
            for (int i = -2; i < idim + 3; ++i) {
                int idx = std::max(0, std::min(nx-1, ng + i));
                for(int m=0; m<4; ++m) {
                    // [修正] T 在 prim(..., 5)
                    double val = (m<3) ? bf.prim(idx,j,k,m+1) : bf.prim(idx,j,k,5);
                    q_line[(i+3)*4 + m] = val;
                }
            }
            interp_line_to_half(4, idim, q_line, uvwt_half, f_l, f_r);

            // 几何量/粘性强制使用中心插值 (0, 0)
            
            // Viscosity
            for (int i = -2; i < idim + 3; ++i) q_line[(i+3)*1 + 0] = bf.mu(std::max(0, std::min(nx-1, ng+i)), j, k);
            interp_line_to_half(1, idim, q_line, vslt1_half, 0, 0); // Force central
            for(int i=0; i<n_face; ++i) vslt2_half[i] = vslt1_half[i] * cp_prl;

            // Vol
            for (int i = -2; i < idim + 3; ++i) q_line[(i+3)*1 + 0] = bf.vol(std::max(0, std::min(nx-1, ng+i)), j, k);
            interp_line_to_half(1, idim, q_line, vol_half, 0, 0); // Force central

            // Metrics
            for (int i = -2; i < idim + 3; ++i) {
                int idx = std::max(0, std::min(nx-1, ng + i));
                for(int m=0; m<9; ++m) q_line[(i+3)*9 + m] = bf.metrics(idx,j,k, m);
            }
            interp_line_to_half(9, idim, q_line, kxyz_half, 0, 0); // Force central

            // Extract Normal (I-dir: kcx, kcy, kcz -> 0,1,2)
            for(int i=0; i<n_face; ++i) {
                normal_vec[i*3+0] = kxyz_half[i*9+0];
                normal_vec[i*3+1] = kxyz_half[i*9+1];
                normal_vec[i*3+2] = kxyz_half[i*9+2];
            }

            // Derivatives (Force central)
            std::vector<double> duvwt_cc_face(n_face * 8);
            for (int i = -2; i < idim + 3; ++i) {
                int src_idx = std::max(ng, std::min(nx-ng-1, ng+i));
                for(int m=0; m<8; ++m) q_line[(i+3)*8+m] = duvwt(src_idx, j, k, m+4);
            }
            interp_line_to_half(8, idim, q_line, duvwt_cc_face, 0, 0); // Force central

            for(int i=0; i<n_face; ++i) {
                int mid_idx = ng + i - 1;
                if(mid_idx>=0 && mid_idx<nx) 
                    for(int m=0; m<4; ++m) duvwt_half[i*12 + m] = duvwt_mid(mid_idx,j,k, m);
                for(int m=0; m<8; ++m) duvwt_half[i*12 + m+4] = duvwt_cc_face[i*8+m];
            }

            // Calc
            std::vector<double> duvwtdxyz(12 * n_face);
            transform_derivs_to_phys(idim, duvwt_half, kxyz_half, vol_half, duvwtdxyz);
            compute_viscous_flux_1d(idim, uvwt_half, duvwtdxyz, vslt1_half, vslt2_half, normal_vec, fv);
            
            // 使用高阶差分
            compute_flux_diff_1d(5, idim, fv, dfv);

            // Update
            for (int i = 0; i < idim; ++i) {
                for (int m = 0; m < 5; ++m) bf.dq(ng+i, j, k, m) -= re_inv * dfv[i*5+m];
            }
        }
    }

    // --- J Direction Loop ---
    int n_face_j = jdim + 1;
    std::vector<double> q_line_j((jdim+6)*9);
    
    uvwt_half.resize(4*n_face_j); duvwt_half.resize(12*n_face_j); kxyz_half.resize(9*n_face_j);
    vol_half.resize(n_face_j); vslt1_half.resize(n_face_j); vslt2_half.resize(n_face_j);
    fv.resize(5*n_face_j); dfv.resize(5*jdim); normal_vec.resize(3*n_face_j);

    for (int k = ng; k < nz - ng; ++k) {
        for (int i = ng; i < nx - ng; ++i) {
            int f_l = (bf.prim(i, ng-3, k, 0) < 0) ? -1 : 0;
            int f_r = (bf.prim(i, ng+jdim+2, k, 0) < 0) ? -1 : 0;

            // UVWT
            for (int j = -2; j < jdim + 3; ++j) {
                int idx = std::max(0, std::min(ny-1, ng + j));
                for(int m=0; m<4; ++m) {
                    // [修正] T 在 prim(..., 5)
                    double val = (m<3) ? bf.prim(i,idx,k,m+1) : bf.prim(i,idx,k,5);
                    q_line_j[(j+3)*4 + m] = val;
                }
            }
            interp_line_to_half(4, jdim, q_line_j, uvwt_half, f_l, f_r);

            // Viscosity & Vol (Central)
            for (int j = -2; j < jdim + 3; ++j) q_line_j[(j+3)*1 + 0] = bf.mu(i, std::max(0, std::min(ny-1, ng+j)), k);
            interp_line_to_half(1, jdim, q_line_j, vslt1_half, 0, 0);
            for(int j=0; j<n_face_j; ++j) vslt2_half[j] = vslt1_half[j] * cp_prl;

            for (int j = -2; j < jdim + 3; ++j) q_line_j[(j+3)*1 + 0] = bf.vol(i, std::max(0, std::min(ny-1, ng+j)), k);
            interp_line_to_half(1, jdim, q_line_j, vol_half, 0, 0);

            // Metrics (Central)
            for (int j = -2; j < jdim + 3; ++j) {
                int idx = std::max(0, std::min(ny-1, ng + j));
                for(int m=0; m<9; ++m) q_line_j[(j+3)*9 + m] = bf.metrics(i,idx,k, m);
            }
            interp_line_to_half(9, jdim, q_line_j, kxyz_half, 0, 0);

            // Extract Normal (J-dir)
            for(int j=0; j<n_face_j; ++j) {
                normal_vec[j*3+0] = kxyz_half[j*9+3];
                normal_vec[j*3+1] = kxyz_half[j*9+4];
                normal_vec[j*3+2] = kxyz_half[j*9+5];
            }

            // Derivatives (Central)
            std::vector<double> duvwt_cc_face(n_face_j * 8);
            for (int j = -2; j < jdim + 3; ++j) {
                int src_idx = std::max(ng, std::min(ny-ng-1, ng+j));
                for(int m=0; m<4; ++m) q_line_j[(j+3)*8+m] = duvwt(i, src_idx, k, m); 
                for(int m=0; m<4; ++m) q_line_j[(j+3)*8+m+4] = duvwt(i, src_idx, k, m+8);
            }
            interp_line_to_half(8, jdim, q_line_j, duvwt_cc_face, 0, 0);

            for(int j=0; j<n_face_j; ++j) {
                int mid_idx = ng + j - 1;
                for(int m=0; m<4; ++m) duvwt_half[j*12 + m] = duvwt_cc_face[j*8+m];
                if(mid_idx>=0 && mid_idx<ny)
                    for(int m=0; m<4; ++m) duvwt_half[j*12 + m+4] = duvwt_mid(i,mid_idx,k, m+4);
                for(int m=0; m<4; ++m) duvwt_half[j*12 + m+8] = duvwt_cc_face[j*8+m+4];
            }

            // Calc
            std::vector<double> duvwtdxyz(12 * n_face_j);
            transform_derivs_to_phys(jdim, duvwt_half, kxyz_half, vol_half, duvwtdxyz);
            compute_viscous_flux_1d(jdim, uvwt_half, duvwtdxyz, vslt1_half, vslt2_half, normal_vec, fv);
            compute_flux_diff_1d(5, jdim, fv, dfv);

            // Update
            for (int j = 0; j < jdim; ++j) {
                for (int m = 0; m < 5; ++m) bf.dq(i, ng+j, k, m) -= re_inv * dfv[j*5+m];
            }
        }
    }

    // --- K Direction Loop ---
    int n_face_k = kdim + 1;
    std::vector<double> q_line_k((kdim+6)*9);
    
    uvwt_half.resize(4*n_face_k); duvwt_half.resize(12*n_face_k); kxyz_half.resize(9*n_face_k);
    vol_half.resize(n_face_k); vslt1_half.resize(n_face_k); vslt2_half.resize(n_face_k);
    fv.resize(5*n_face_k); dfv.resize(5*kdim); normal_vec.resize(3*n_face_k);

    for (int j = ng; j < ny - ng; ++j) {
        for (int i = ng; i < nx - ng; ++i) {
            int f_l = (bf.prim(i, j, ng-3, 0) < 0) ? -1 : 0;
            int f_r = (bf.prim(i, j, ng+kdim+2, 0) < 0) ? -1 : 0;

            // UVWT
            for (int k = -2; k < kdim + 3; ++k) {
                int idx = std::max(0, std::min(nz-1, ng + k));
                for(int m=0; m<4; ++m) {
                    // [修正] T 在 prim(..., 5)
                    double val = (m<3) ? bf.prim(i,j,idx,m+1) : bf.prim(i,j,idx,5);
                    q_line_k[(k+3)*4 + m] = val;
                }
            }
            interp_line_to_half(4, kdim, q_line_k, uvwt_half, f_l, f_r);

            // Visc/Vol (Central)
            for (int k = -2; k < kdim + 3; ++k) q_line_k[(k+3)*1 + 0] = bf.mu(i, j, std::max(0, std::min(nz-1, ng+k)));
            interp_line_to_half(1, kdim, q_line_k, vslt1_half, 0, 0);
            for(int k=0; k<n_face_k; ++k) vslt2_half[k] = vslt1_half[k] * cp_prl;

            for (int k = -2; k < kdim + 3; ++k) q_line_k[(k+3)*1 + 0] = bf.vol(i, j, std::max(0, std::min(nz-1, ng+k)));
            interp_line_to_half(1, kdim, q_line_k, vol_half, 0, 0);

            // Metrics (Central)
            for (int k = -2; k < kdim + 3; ++k) {
                int idx = std::max(0, std::min(nz-1, ng + k));
                for(int m=0; m<9; ++m) q_line_k[(k+3)*9 + m] = bf.metrics(i,j,idx, m);
            }
            interp_line_to_half(9, kdim, q_line_k, kxyz_half, 0, 0);

            // Extract Normal (K-dir)
            for(int k=0; k<n_face_k; ++k) {
                normal_vec[k*3+0] = kxyz_half[k*9+6];
                normal_vec[k*3+1] = kxyz_half[k*9+7];
                normal_vec[k*3+2] = kxyz_half[k*9+8];
            }

            // Derivatives (Central)
            std::vector<double> duvwt_cc_face(n_face_k * 8);
            for (int k = -2; k < kdim + 3; ++k) {
                int src_idx = std::max(ng, std::min(nz-ng-1, ng+k));
                for(int m=0; m<8; ++m) q_line_k[(k+3)*8+m] = duvwt(i, j, src_idx, m); 
            }
            interp_line_to_half(8, kdim, q_line_k, duvwt_cc_face, 0, 0);

            for(int k=0; k<n_face_k; ++k) {
                int mid_idx = ng + k - 1;
                for(int m=0; m<8; ++m) duvwt_half[k*12 + m] = duvwt_cc_face[k*8+m];
                if(mid_idx>=0 && mid_idx<nz)
                    for(int m=0; m<4; ++m) duvwt_half[k*12 + m+8] = duvwt_mid(i,j,mid_idx, m+8);
            }

            // Calc
            std::vector<double> duvwtdxyz(12 * n_face_k);
            transform_derivs_to_phys(kdim, duvwt_half, kxyz_half, vol_half, duvwtdxyz);
            compute_viscous_flux_1d(kdim, uvwt_half, duvwtdxyz, vslt1_half, vslt2_half, normal_vec, fv);
            compute_flux_diff_1d(5, kdim, fv, dfv);

            // Update
            for (int k = 0; k < kdim; ++k) {
                for (int m = 0; m < 5; ++m) bf.dq(i, j, ng+k, m) -= re_inv * dfv[k*5+m];
            }
        }
    }
}

void FluxComputer::compute_viscous_rhs(orion::preprocess::FlowFieldSet& fs, 
                                       const orion::core::Params& params)
{
    if (params.flowtype.nvis != 1) return;
    int ng = fs.ng;
    for (int nb : fs.local_block_ids) {
        compute_viscous_flux_block(fs.blocks[nb], params, ng);
    }
}

} // namespace orion::solver