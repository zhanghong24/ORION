#include "solver/HaloExchanger.hpp"
#include "core/Runtime.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <list>
#include <mpi.h>

namespace orion::solver {

// ===========================================================================
// 辅助函数 1: 获取发送窗口 (复刻 Fortran 逻辑: Ghost Layers + 切向扩展)
// ===========================================================================
static void get_fortran_window(const orion::bc::BCRegion& reg, 
                               const std::vector<std::size_t>& dims, 
                               int layers, 
                               int (&beg)[3], int (&end)[3]) 
{
    // 1. 基础面定义 (s_st/s_ed 是 1-based, 转为 0-based)
    for(int m=0; m<3; ++m) {
        beg[m] = std::min(std::abs(reg.s_st[m]), std::abs(reg.s_ed[m])) - 1;
        end[m] = std::max(std::abs(reg.s_st[m]), std::abs(reg.s_ed[m])) - 1;
    }

    int dir = reg.s_nd - 1; // 法向 (0,1,2)
    int inrout = reg.s_lr;  // -1(Min面) 或 1(Max面)

    // 2. 法向扩展 (Ghost Layers) - 如果 layers > 0
    if (layers > 0) {
        if (inrout == 1) {
            beg[dir] -= layers; // 向内(负方向)扩展
        } else {
            end[dir] += layers; // 向内(正方向)扩展
        }
    }

    // 3. 切向扩展 (Tangential Extension) - 仅在有 ghost 扩展时才进行切向扩展
    // Fortran 逻辑: ibeg = ibeg - 4... 然后 if(ibeg>1) ibeg-1
    // 如果 layers=0 (DQ通信)，Fortran 没有做切向扩展。
    if (layers > 0) {
        int t_dirs[2];
        if (dir == 0) { t_dirs[0]=1; t_dirs[1]=2; }
        else if (dir == 1) { t_dirs[0]=2; t_dirs[1]=0; }
        else { t_dirs[0]=0; t_dirs[1]=1; }

        for (int t : t_dirs) {
            if (beg[t] > 0) beg[t] -= 1;
            if (end[t] < (int)dims[t] - 1) end[t] += 1; 
        }
    }
}

// ===========================================================================
// 辅助函数 2: 推算对方发送窗口的尺寸 (用于解包时的 Stride 计算)
// ===========================================================================
static void get_remote_window_dims(const orion::bc::BCRegion& reg, 
                                   int layers,
                                   std::array<int, 3>& dims_out) 
{
    orion::bc::BCRegion fake_remote;
    fake_remote.s_st = reg.t_st; 
    fake_remote.s_ed = reg.t_ed;
    fake_remote.s_nd = reg.t_nd;
    fake_remote.s_lr = reg.t_lr;
    
    std::vector<std::size_t> dummy_dims = {999999, 999999, 999999}; 
    int r_beg[3], r_end[3];
    
    get_fortran_window(fake_remote, dummy_dims, layers, r_beg, r_end);

    dims_out[0] = r_end[0] - r_beg[0] + 1;
    dims_out[1] = r_end[1] - r_beg[1] + 1;
    dims_out[2] = r_end[2] - r_beg[2] + 1;
}

// ===========================================================================
// 主函数: 交换并应用边界条件 (Primitive Variables)
// ===========================================================================
void HaloExchanger::exchange_bc(orion::bc::BCData& bc, orion::preprocess::FlowFieldSet& fs) {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int ng = fs.ng;

    // 使用 list 存储 buffer，保证指针有效性
    std::list<std::vector<double>> send_buffers; 
    std::vector<MPI_Request> send_reqs;

    // -----------------------------------------------------------------------
    // Loop 1: 发送所有数据 (Non-blocking Send)
    // -----------------------------------------------------------------------
    for (int nb_idx : fs.local_block_ids) {
        auto& bcb = bc.block_bc[nb_idx];
        auto& bf = fs.blocks[nb_idx];
        
        const auto& dims = bf.prim.dims(); 
        int dim_i = (int)dims[0];
        int dim_j = (int)dims[1];
        int dim_k = (int)dims[2];

        for (auto& reg : bcb.regions) {
            if (!reg.is_connect()) continue;

            int neighbor_pid = bc.block_pid[reg.nbt - 1] - 1;
            int send_tag = reg.nbt * 1000 + (nb_idx + 1); 

            // 1. 计算窗口 (带 Ghost 扩展)
            int beg[3], end[3];
            get_fortran_window(reg, dims, ng, beg, end);

            int ni = end[0] - beg[0] + 1;
            int nj = end[1] - beg[1] + 1;
            int nk = end[2] - beg[2] + 1;
            int nvar = 5; 

            // 2. 打包数据
            send_buffers.emplace_back();
            auto& sbuf = send_buffers.back();
            sbuf.reserve(ni * nj * nk * nvar);

            for (int k = beg[2]; k <= end[2]; ++k) {
                for (int j = beg[1]; j <= end[1]; ++j) {
                    for (int i = beg[0]; i <= end[0]; ++i) {
                        
                        int idx_i = i + ng;
                        int idx_j = j + ng;
                        int idx_k = k + ng;

                        if (idx_i >= 0 && idx_i < dim_i &&
                            idx_j >= 0 && idx_j < dim_j &&
                            idx_k >= 0 && idx_k < dim_k) 
                        {
                            for (int m = 0; m < nvar; ++m) {
                                sbuf.push_back(bf.prim(idx_i, idx_j, idx_k, m));
                            }
                        } else {
                            for (int m = 0; m < nvar; ++m) sbuf.push_back(0.0);
                        }
                    }
                }
            }

            // 3. 非阻塞发送
            MPI_Request req;
            MPI_Isend(sbuf.data(), (int)sbuf.size(), MPI_DOUBLE, neighbor_pid, 
                      send_tag, MPI_COMM_WORLD, &req);
            send_reqs.push_back(req);
        }
    }

    // -----------------------------------------------------------------------
    // Loop 2: 接收所有数据 (Blocking Probe & Recv)
    // -----------------------------------------------------------------------
    for (int nb_idx : fs.local_block_ids) {
        auto& bcb = bc.block_bc[nb_idx];
        for (auto& reg : bcb.regions) {
            if (!reg.is_connect()) continue;

            int neighbor_pid = bc.block_pid[reg.nbt - 1] - 1;
            int recv_tag = (nb_idx + 1) * 1000 + reg.nbt;

            MPI_Status status;
            MPI_Probe(neighbor_pid, recv_tag, MPI_COMM_WORLD, &status);

            int count;
            MPI_Get_count(&status, MPI_DOUBLE, &count);

            reg.qpvpack.resize(count);

            MPI_Recv(reg.qpvpack.data(), count, MPI_DOUBLE, neighbor_pid, 
                     recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            get_remote_window_dims(reg, ng, reg.pack_dims);
        }
    }

    // -----------------------------------------------------------------------
    // Wait: 确保所有发送完成
    // -----------------------------------------------------------------------
    if (!send_reqs.empty()) {
        MPI_Waitall((int)send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
    }

    // -----------------------------------------------------------------------
    // Phase 3: 应用 (Unpack with Topology Mapping)
    // -----------------------------------------------------------------------
    for (int nb_idx : fs.local_block_ids) {
        auto& bcb = bc.block_bc[nb_idx];
        auto& bf = fs.blocks[nb_idx];
        
        const auto& dims = bf.prim.dims(); 
        int dim_i = (int)dims[0];
        int dim_j = (int)dims[1];
        int dim_k = (int)dims[2];

        for (auto& reg : bcb.regions) {
            if (!reg.is_connect()) continue;
            if (reg.qpvpack.empty()) continue;

            int ni_s = reg.pack_dims[0];
            int nj_s = reg.pack_dims[1];

            // 获取 Sender 的 interior 起始坐标
            int r_beg[3], r_end[3];
            std::vector<std::size_t> dummy = {999999, 999999, 999999};
            orion::bc::BCRegion fake; 
            fake.s_st=reg.t_st; fake.s_ed=reg.t_ed; 
            fake.s_nd=reg.t_nd; fake.s_lr=reg.t_lr;
            get_fortran_window(fake, dummy, ng, r_beg, r_end);

            // 遍历本地 Face
            int s_st[3] = {std::abs(reg.s_st[0])-1, std::abs(reg.s_st[1])-1, std::abs(reg.s_st[2])-1};
            int s_ed[3] = {std::abs(reg.s_ed[0])-1, std::abs(reg.s_ed[1])-1, std::abs(reg.s_ed[2])-1};
            
            int i_min = std::min(s_st[0], s_ed[0]); int i_max = std::max(s_st[0], s_ed[0]);
            int j_min = std::min(s_st[1], s_ed[1]); int j_max = std::max(s_st[1], s_ed[1]);
            int k_min = std::min(s_st[2], s_ed[2]); int k_max = std::max(s_st[2], s_ed[2]);

            int dim_i_loc = reg.map_dims[0];
            int dim_j_loc = reg.map_dims[1];

            int s_lr3d[3] = {reg.s_lr3d[0], reg.s_lr3d[1], reg.s_lr3d[2]};
            int t_lr3d[3] = {reg.t_lr3d[0], reg.t_lr3d[1], reg.t_lr3d[2]};

            for (int k = k_min; k <= k_max; ++k) {
                for (int j = j_min; j <= j_max; ++j) {
                    for (int i = i_min; i <= i_max; ++i) {
                        
                        int l_i = i - i_min;
                        int l_j = j - j_min;
                        int l_k = k - k_min;
                        
                        if (l_i >= dim_i_loc || l_j >= dim_j_loc) continue;

                        int map_idx = (l_k * dim_j_loc + l_j) * dim_i_loc + l_i;
                        if (map_idx < 0 || map_idx >= (int)reg.image.size()) continue;

                        int it_face = reg.image[map_idx] - 1; 
                        int jt_face = reg.jmage[map_idx] - 1;
                        int kt_face = reg.kmage[map_idx] - 1;

                        // 逐层扩展 (Ghost Layers)
                        for (int l = 0; l < ng; ++l) { // 0-based layer
                            int dist = l + 1;
                            
                            int is = i + dist * s_lr3d[0];
                            int js = j + dist * s_lr3d[1];
                            int ks = k + dist * s_lr3d[2];

                            int it_global = it_face - dist * t_lr3d[0];
                            int jt_global = jt_face - dist * t_lr3d[1];
                            int kt_global = kt_face - dist * t_lr3d[2];

                            int it = it_global - r_beg[0];
                            int jt = jt_global - r_beg[1];
                            int kt = kt_global - r_beg[2];

                            if (it < 0 || it >= ni_s || jt < 0 || jt >= nj_s) continue;
                            
                            size_t offset = ((size_t)kt * nj_s + jt) * ni_s + it;
                            offset *= 5;

                            if (offset + 4 < reg.qpvpack.size()) {
                                int idx_i = is + ng;
                                int idx_j = js + ng;
                                int idx_k = ks + ng;

                                if (idx_i >= 0 && idx_i < dim_i &&
                                    idx_j >= 0 && idx_j < dim_j &&
                                    idx_k >= 0 && idx_k < dim_k) 
                                {
                                    for (int m = 0; m < 5; ++m) {
                                        bf.prim(idx_i, idx_j, idx_k, m) = reg.qpvpack[offset + m];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 新增函数: 交换并平均接口残差 (复刻 Fortran exchange_bc_dq_vol + boundary_match_dq_2pm)
// ===========================================================================
void HaloExchanger::average_interface_residuals(orion::bc::BCData& bc, orion::preprocess::FlowFieldSet& fs) {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    int ng = fs.ng;

    std::list<std::vector<double>> send_buffers; 
    std::vector<MPI_Request> send_reqs;

    // --- Phase 1: Pack & Send (Send DQ/VOL) ---
    // 仅发送 Face 上的数据，不扩展 Ghost
    for (int nb_idx : fs.local_block_ids) {
        auto& bcb = bc.block_bc[nb_idx];
        auto& bf = fs.blocks[nb_idx];

        for (auto& reg : bcb.regions) {
            if (!reg.is_connect()) continue;

            int neighbor_pid = bc.block_pid[reg.nbt - 1] - 1;
            int send_tag = reg.nbt * 2000 + (nb_idx + 1); 

            // 1. 窗口: 仅 Face (layers=0)
            int beg[3], end[3];
            const auto& dims = bf.dq.dims(); // 其实 dq 和 prim 维度一样
            get_fortran_window(reg, dims, 0, beg, end);

            int ni = end[0] - beg[0] + 1;
            int nj = end[1] - beg[1] + 1;
            int nk = end[2] - beg[2] + 1;
            int nvar = 5;

            // 2. 打包 dq/vol
            send_buffers.emplace_back();
            auto& sbuf = send_buffers.back();
            sbuf.reserve(ni * nj * nk * nvar);

            for (int k = beg[2]; k <= end[2]; ++k) {
                for (int j = beg[1]; j <= end[1]; ++j) {
                    for (int i = beg[0]; i <= end[0]; ++i) {
                        int idx_i = i + ng;
                        int idx_j = j + ng;
                        int idx_k = k + ng;
                        
                        double vol = bf.vol(idx_i, idx_j, idx_k);
                        if (vol < 1e-30) vol = 1e-30;

                        for (int m = 0; m < nvar; ++m) {
                            sbuf.push_back(bf.dq(idx_i, idx_j, idx_k, m) / vol);
                        }
                    }
                }
            }

            MPI_Request req;
            MPI_Isend(sbuf.data(), (int)sbuf.size(), MPI_DOUBLE, neighbor_pid, 
                      send_tag, MPI_COMM_WORLD, &req);
            send_reqs.push_back(req);
        }
    }

    // --- Phase 2: Recv & Average (boundary_match_dq_2pm) ---
    for (int nb_idx : fs.local_block_ids) {
        auto& bcb = bc.block_bc[nb_idx];
        auto& bf = fs.blocks[nb_idx];

        for (auto& reg : bcb.regions) {
            if (!reg.is_connect()) continue;

            int neighbor_pid = bc.block_pid[reg.nbt - 1] - 1;
            int recv_tag = (nb_idx + 1) * 2000 + reg.nbt;

            MPI_Status status;
            MPI_Probe(neighbor_pid, recv_tag, MPI_COMM_WORLD, &status);

            int count;
            MPI_Get_count(&status, MPI_DOUBLE, &count);

            reg.qpvpack.resize(count); // 重用 qpvpack

            MPI_Recv(reg.qpvpack.data(), count, MPI_DOUBLE, neighbor_pid, 
                     recv_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // 计算远程 Face 尺寸 (无 Ghost) 用于解包
            int r_beg[3], r_end[3];
            std::array<int, 3> r_dims_arr;
            get_remote_window_dims(reg, 0, r_dims_arr); // layers=0
            int ni_s = r_dims_arr[0];
            
            // 为了计算 r_beg (Sender offset)，我们还是需要调用一次 get_fortran_window
            // 因为 get_remote_window_dims 只返回了尺寸
            std::vector<std::size_t> dummy = {999999, 999999, 999999};
            orion::bc::BCRegion fake; 
            fake.s_st=reg.t_st; fake.s_ed=reg.t_ed; 
            fake.s_nd=reg.t_nd; fake.s_lr=reg.t_lr;
            get_fortran_window(fake, dummy, 0, r_beg, r_end);

            // 遍历本地 Face 进行平均
            // 窗口逻辑同发送 (layers=0)
            int s_beg[3], s_end[3];
            const auto& dims = bf.dq.dims();
            get_fortran_window(reg, dims, 0, s_beg, s_end);

            int dim_i_loc = reg.map_dims[0];
            int dim_j_loc = reg.map_dims[1];

            for (int k = s_beg[2]; k <= s_end[2]; ++k) {
                for (int j = s_beg[1]; j <= s_end[1]; ++j) {
                    for (int i = s_beg[0]; i <= s_end[0]; ++i) {
                        
                        int l_i = i - s_beg[0];
                        int l_j = j - s_beg[1];
                        int l_k = k - s_beg[2];

                        if (l_i >= dim_i_loc || l_j >= dim_j_loc) continue;
                        
                        int map_idx = (l_k * dim_j_loc + l_j) * dim_i_loc + l_i;
                        if (map_idx < 0 || map_idx >= (int)reg.image.size()) continue;

                        int it_global = reg.image[map_idx] - 1; 
                        int jt_global = reg.jmage[map_idx] - 1;
                        int kt_global = reg.kmage[map_idx] - 1;

                        int it = it_global - r_beg[0];
                        int jt = jt_global - r_beg[1];
                        int kt = kt_global - r_beg[2];

                        // 发送方 buffer 维度
                        // 注意 get_remote_window_dims 返回的是 dimensions，不是 stride
                        // Stride 应该基于 dimensions 计算
                        // 远程 buffer 是 ni_s * nj_s * nk_s * 5
                        // 这里的 ni_s 来自 r_dims_arr[0]
                        int nj_s = r_dims_arr[1];

                        if (it < 0 || it >= ni_s || jt < 0 || jt >= nj_s) continue;

                        size_t offset = ((size_t)kt * nj_s + jt) * ni_s + it;
                        offset *= 5;

                        if (offset + 4 < reg.qpvpack.size()) {
                            int idx_i = i + ng;
                            int idx_j = j + ng;
                            int idx_k = k + ng;
                            
                            double vol_self = bf.vol(idx_i, idx_j, idx_k);

                            for (int m = 0; m < 5; ++m) {
                                double val_neigh = reg.qpvpack[offset + m];
                                // Fortran: dq = 0.5 * (dq + dq_neigh * vol_self / vol_neigh)
                                // val_neigh = dq_neigh / vol_neigh
                                bf.dq(idx_i, idx_j, idx_k, m) = 
                                    0.5 * (bf.dq(idx_i, idx_j, idx_k, m) + val_neigh * vol_self);
                            }
                        }
                    }
                }
            }
        }
    }

    if (!send_reqs.empty()) {
        MPI_Waitall((int)send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
    }
}

} // namespace orion::solver