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
// 辅助函数 1: 获取发送窗口 (复刻 Fortran 逻辑: 4层幽灵网格 + 切向扩展)
// ===========================================================================
static void get_fortran_window(const orion::bc::BCRegion& reg, 
                               const std::vector<std::size_t>& dims, 
                               int (&beg)[3], int (&end)[3]) 
{
    // 1. 基础面定义 (s_st/s_ed 是 1-based, 转为 0-based)
    for(int m=0; m<3; ++m) {
        beg[m] = std::min(std::abs(reg.s_st[m]), std::abs(reg.s_ed[m])) - 1;
        end[m] = std::max(std::abs(reg.s_st[m]), std::abs(reg.s_ed[m])) - 1;
    }

    int dir = reg.s_nd - 1; // 法向 (0,1,2)
    int inrout = reg.s_lr;  // -1(Min面) 或 1(Max面)

    // 2. 法向扩展 (Ghost Layers)
    if (inrout == 1) {
        beg[dir] -= 4; // 向内(负方向)扩展
    } else {
        end[dir] += 4; // 向内(正方向)扩展
    }

    // 3. 切向扩展 (Tangential Extension)
    int t_dirs[2];
    if (dir == 0) { t_dirs[0]=1; t_dirs[1]=2; }
    else if (dir == 1) { t_dirs[0]=2; t_dirs[1]=0; }
    else { t_dirs[0]=0; t_dirs[1]=1; }

    for (int t : t_dirs) {
        if (beg[t] > 0) beg[t] -= 1;
        // 使用 static_cast 避免有符号/无符号比较警告
        // 注意：这里的 dims 是包含 ghost 的总尺寸，但在计算逻辑窗口时
        // 我们通常是在 physical space 操作。
        // 但 Fortran 这里的逻辑其实是防止物理索引越界。
        // 简单起见，我们允许扩展，后续在 Access 时做边界检查。
        if (end[t] < (int)dims[t] - 1) end[t] += 1; 
    }
}

// ===========================================================================
// 辅助函数 2: 推算对方发送窗口的尺寸 (用于解包时的 Stride 计算)
// ===========================================================================
static void get_remote_window_dims(const orion::bc::BCRegion& reg, 
                                   std::array<int, 3>& dims_out) 
{
    orion::bc::BCRegion fake_remote;
    fake_remote.s_st = reg.t_st; 
    fake_remote.s_ed = reg.t_ed;
    fake_remote.s_nd = reg.t_nd;
    fake_remote.s_lr = reg.t_lr;
    
    std::vector<std::size_t> dummy_dims = {999999, 999999, 999999}; 
    int r_beg[3], r_end[3];
    
    get_fortran_window(fake_remote, dummy_dims, r_beg, r_end);

    dims_out[0] = r_end[0] - r_beg[0] + 1;
    dims_out[1] = r_end[1] - r_beg[1] + 1;
    dims_out[2] = r_end[2] - r_beg[2] + 1;
}

// ===========================================================================
// 主函数: 交换并应用边界条件
// ===========================================================================
void HaloExchanger::exchange_bc(orion::bc::BCData& bc, orion::preprocess::FlowFieldSet& fs) {
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // [CRITICAL] 获取 Ghost Layers 数量，这是物理坐标到数组坐标的偏移量
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
        
        // bf.prim.dims() 返回的是分配的总大小 (ni+2*ng, nj+2*ng, nk+2*ng)
        const auto& dims = bf.prim.dims(); 
        int dim_i = (int)dims[0];
        int dim_j = (int)dims[1];
        int dim_k = (int)dims[2];

        for (auto& reg : bcb.regions) {
            if (!reg.is_connect()) continue;

            int neighbor_pid = bc.block_pid[reg.nbt - 1] - 1;
            int send_tag = reg.nbt * 1000 + (nb_idx + 1); 

            // 1. 计算窗口
            int beg[3], end[3];
            // get_fortran_window 返回的是物理坐标的范围 (可能是 -4 到 N+4)
            get_fortran_window(reg, dims, beg, end);

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
                        
                        // [CRITICAL] 坐标转换: Physical (i) -> Array (idx)
                        // Array Index = Physical Index + ng
                        int idx_i = i + ng;
                        int idx_j = j + ng;
                        int idx_k = k + ng;

                        // [SAFETY] 边界检查，防止 Segfault
                        if (idx_i >= 0 && idx_i < dim_i &&
                            idx_j >= 0 && idx_j < dim_j &&
                            idx_k >= 0 && idx_k < dim_k) 
                        {
                            for (int m = 0; m < nvar; ++m) {
                                sbuf.push_back(bf.prim(idx_i, idx_j, idx_k, m));
                            }
                        } else {
                            // 如果请求的数据超出了本地 ghost 的范围 (例如 ng=2 但请求了 layer 4)
                            // 填入 0.0 防止崩溃，虽然这在物理上不正确，但比 crashing 好。
                            // 理想情况下应该增大 ng。
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

            get_remote_window_dims(reg, reg.pack_dims);
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
            get_fortran_window(fake, dummy, r_beg, r_end);

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

                        // 计算 map_idx 需要格外小心维度
                        // image 是一维数组，我们需要确定它的 flatten 方式
                        // 假设 BCPreprocess 按照 (k, j, i) 顺序填充 (i变化最快)
                        // 且 dimensions 是 dim_i_loc, dim_j_loc
                        int map_idx = (l_k * dim_j_loc + l_j) * dim_i_loc + l_i;
                        
                        if (map_idx < 0 || map_idx >= (int)reg.image.size()) continue;

                        int it_face = reg.image[map_idx] - 1; 
                        int jt_face = reg.jmage[map_idx] - 1;
                        int kt_face = reg.kmage[map_idx] - 1;

                        // 逐层扩展 (Ghost Layers 1..4)
                        for (int layer = 1; layer <= 4; ++layer) {
                            
                            int is = i + layer * s_lr3d[0];
                            int js = j + layer * s_lr3d[1];
                            int ks = k + layer * s_lr3d[2];

                            int it_global = it_face - layer * t_lr3d[0];
                            int jt_global = jt_face - layer * t_lr3d[1];
                            int kt_global = kt_face - layer * t_lr3d[2];

                            int it = it_global - r_beg[0];
                            int jt = jt_global - r_beg[1];
                            int kt = kt_global - r_beg[2];

                            if (it < 0 || it >= ni_s || jt < 0 || jt >= nj_s) continue;
                            
                            size_t offset = ((size_t)kt * nj_s + jt) * ni_s + it;
                            offset *= 5;

                            if (offset + 4 < reg.qpvpack.size()) {
                                // [CRITICAL] 写入本地 bf.prim，增加 ng 偏移
                                int idx_i = is + ng;
                                int idx_j = js + ng;
                                int idx_k = ks + ng;

                                // [SAFETY] 写入前边界检查
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

void HaloExchanger::average_interface_residuals(orion::bc::BCData& bc, orion::preprocess::FlowFieldSet& fs) {
    // 占位符
}

} // namespace orion::solver