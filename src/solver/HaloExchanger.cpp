#include "solver/HaloExchanger.hpp"
#include "core/Runtime.hpp"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <mpi.h>

namespace orion::solver {

static inline int get_cyc(int idir, int shift) {
    return (idir + shift) % 3;
}

// 获取通信窗口 (用于 exchange_bc)
void HaloExchanger::get_window(const CommTask& task, 
                               const orion::preprocess::BlockField& bf,
                               int ng, bool is_ghost,
                               int& i_start, int& i_end,
                               int& j_start, int& j_end,
                               int& k_start, int& k_end)
{
    std::array<int,3> beg = task.s_st;
    std::array<int,3> end = task.s_ed;
    
    int dir = task.dir - 1;   // 法向: 0, 1, 2
    int inrout = task.inrout; // 1: Max Face, -1: Min Face

    // [防御性检查]
    if (dir < 0 || dir > 2) {
        std::cerr << "FATAL: HaloExchanger invalid direction! Block " << task.global_nb 
                  << " Region " << task.nr << " s_nd=" << task.dir << "\n";
        std::exit(-1);
    }

    if (!bf.allocated()) {
        std::cerr << "FATAL: Accessing unallocated block " << (task.global_nb + 1) 
                  << " in HaloExchanger! Check block distribution.\n";
        std::exit(-1);
    }

    // 法向偏移
    if (!is_ghost) {
        // [SEND] 读取内部数据
        if (inrout == 1) { 
            beg[dir] = beg[dir] - ng + 1;
        } else {
            // Min Face or internal cut
            end[dir] = end[dir] + ng - 1; 
        }
    } else {
        // [RECV] 写入幽灵网格
        if (inrout == 1) { 
            beg[dir] = beg[dir] + 1;
            end[dir] = end[dir] + ng;
        } else { 
            beg[dir] = beg[dir] - ng;
            end[dir] = end[dir] - 1;
        }
    }

    // 切向扩展
    int p_dims[3];
    const auto& dims = bf.vol.dims();
    if (dims.size() < 3) {
         std::cerr << "FATAL: Block volume dims invalid (" << dims.size() << ") for Block " << (task.global_nb+1) << "\n";
         std::exit(-1);
    }
    p_dims[0] = (int)dims[0] - 2 * ng;
    p_dims[1] = (int)dims[1] - 2 * ng;
    p_dims[2] = (int)dims[2] - 2 * ng;

    for (int m = 1; m <= 2; ++m) {
        int t_dir = get_cyc(dir, m);
        if (beg[t_dir] > 1) {
            beg[t_dir] -= 1;
        }
        if (end[t_dir] < p_dims[t_dir]) {
            end[t_dir] += 1;
        }
    }

    i_start = beg[0] - 1 + ng;
    i_end   = end[0] - 1 + ng;
    j_start = beg[1] - 1 + ng;
    j_end   = end[1] - 1 + ng;
    k_start = beg[2] - 1 + ng;
    k_end   = end[2] - 1 + ng;
}

// ===========================================================================
// 1. 原始变量交换 (Exchange BC)
// ===========================================================================
void HaloExchanger::exchange_bc(const orion::bc::BCData& bc,
                                orion::preprocess::FlowFieldSet& fs)
{
    int myrank = orion::core::Runtime::rank();
    int ng = fs.ng; 
    
    reqs_.clear();
    send_buffers_.clear();
    recv_buffers_.clear();

    std::vector<CommTask> send_tasks;
    std::vector<CommTask> recv_tasks;
    std::vector<CommTask> local_copy_tasks;

    // 1. Identify Tasks
    for (int nb_idx = 0; nb_idx < (int)fs.local_block_ids.size(); ++nb_idx) {
        int nb = fs.local_block_ids[nb_idx];
        const auto& bcb = bc.block_bc[nb];

        for (int nr = 0; nr < bcb.nregions; ++nr) {
            const auto& reg = bcb.regions[nr];
            
            if (reg.bctype < 0) { 
                int target_nb = reg.nbt - 1; 
                if (target_nb < 0 || target_nb >= (int)bc.block_pid.size()) {
                    std::cerr << "FATAL: Invalid target_nb " << target_nb << " in Block " << nb+1 << "\n";
                    std::exit(-1); 
                }
                int target_rank = bc.block_pid[target_nb] - 1;

                CommTask task;
                task.local_nb_idx = nb_idx;
                task.global_nb = nb;
                task.nr = nr;
                task.target_nb = target_nb;
                task.target_rank = target_rank;
                task.target_nr = reg.ibcwin; 
                
                task.s_st = reg.s_st;
                task.s_ed = reg.s_ed;
                task.dir = reg.s_nd;
                task.inrout = reg.s_lr;

                if (target_rank == myrank) {
                    local_copy_tasks.push_back(task);
                } else {
                    send_tasks.push_back(task);
                    recv_tasks.push_back(task);
                }
            }
        }
    }

    auto GenerateTag = [](int dest_nb, int dest_nr_1based) {
        return 10000 + dest_nb * 1000 + dest_nr_1based;
    };

    // 2. Irecv
    for (const auto& task : recv_tasks) {
        auto& bf = fs.blocks[task.global_nb];
        
        int is, ie, js, je, ks, ke;
        get_window(task, bf, ng, true, is, ie, js, je, ks, ke);
        
        long long n_cells = (long long)(ie - is + 1) * (je - js + 1) * (ke - ks + 1);
        if (n_cells <= 0) continue; 
        
        size_t count = (size_t)n_cells * 5;
        try {
            std::vector<double> buf(count);
            recv_buffers_.push_back(std::move(buf));
        } catch (const std::exception& e) {
            std::cerr << "FATAL: Alloc failed for Recv. Rank=" << myrank 
                      << " Size=" << count << " Err=" << e.what() << "\n";
            std::exit(-1);
        }

        double* ptr = recv_buffers_.back().data();
        int tag = GenerateTag(task.global_nb, task.nr + 1);
        
        MPI_Request req;
        MPI_Irecv(ptr, count, MPI_DOUBLE, task.target_rank, tag, MPI_COMM_WORLD, &req);
        reqs_.push_back(req);
    }

    // 3. Isend
    for (const auto& task : send_tasks) {
        const auto& bf = fs.blocks[task.global_nb];
        
        int is, ie, js, je, ks, ke;
        get_window(task, bf, ng, false, is, ie, js, je, ks, ke);
        
        long long n_cells = (long long)(ie - is + 1) * (je - js + 1) * (ke - ks + 1);
        if (n_cells <= 0) continue; // Skip degenerated faces

        std::vector<double> buf;
        buf.reserve(n_cells * 5);
        
        for (int k = ks; k <= ke; ++k) {
            for (int j = js; j <= je; ++j) {
                for (int i = is; i <= ie; ++i) {
                    for(int m=0; m<5; ++m) buf.push_back(bf.prim(i, j, k, m));
                }
            }
        }
        
        send_buffers_.push_back(std::move(buf));
        double* ptr = send_buffers_.back().data();
        int tag = GenerateTag(task.target_nb, task.target_nr);
        
        MPI_Request req;
        MPI_Isend(ptr, (int)send_buffers_.back().size(), MPI_DOUBLE, task.target_rank, tag, MPI_COMM_WORLD, &req);
        reqs_.push_back(req);
    }

    if (!reqs_.empty()) {
        MPI_Waitall((int)reqs_.size(), reqs_.data(), MPI_STATUSES_IGNORE);
    }

    // 5. Unpack
    for (size_t i = 0; i < recv_tasks.size(); ++i) {
        const auto& task = recv_tasks[i];
        if (i >= recv_buffers_.size()) continue;

        auto& bf = fs.blocks[task.global_nb];
        const double* ptr = recv_buffers_[i].data();
        
        int is, ie, js, je, ks, ke;
        get_window(task, bf, ng, true, is, ie, js, je, ks, ke);
        
        int idx = 0;
        for (int k = ks; k <= ke; ++k) {
            for (int j = js; j <= je; ++j) {
                for (int i = is; i <= ie; ++i) {
                    for(int m=0; m<5; ++m) bf.prim(i, j, k, m) = ptr[idx++];
                }
            }
        }
    }

    // 6. Local Copy
    for (const auto& task : local_copy_tasks) {
        auto& bf_src = fs.blocks[task.global_nb];
        
        int s_is, s_ie, s_js, s_je, s_ks, s_ke;
        get_window(task, bf_src, ng, false, s_is, s_ie, s_js, s_je, s_ks, s_ke);

        auto& bf_dst = fs.blocks[task.target_nb];
        if (!bf_dst.allocated()) {
             std::cerr << "FATAL: Local copy target block " << (task.target_nb+1) << " unallocated!\n";
             std::exit(-1);
        }
        
        int t_nr = task.target_nr - 1; 
        const auto& bcb_dst = bc.block_bc[task.target_nb];
        const auto& reg_dst = bcb_dst.regions[t_nr];
        
        CommTask t_recv_mock;
        t_recv_mock.s_st = reg_dst.s_st;
        t_recv_mock.s_ed = reg_dst.s_ed;
        t_recv_mock.dir  = reg_dst.s_nd;
        t_recv_mock.inrout = reg_dst.s_lr;
        t_recv_mock.global_nb = task.target_nb; 
        
        int d_is, d_ie, d_js, d_je, d_ks, d_ke;
        get_window(t_recv_mock, bf_dst, ng, true, d_is, d_ie, d_js, d_je, d_ks, d_ke);

        int ni = s_ie - s_is + 1;
        int nj = s_je - s_js + 1;
        int nk = s_ke - s_ks + 1;
        
        for (int k = 0; k < nk; ++k) {
            for (int j = 0; j < nj; ++j) {
                for (int i = 0; i < ni; ++i) {
                    for (int v = 0; v < 5; ++v) {
                        bf_dst.prim(d_is+i, d_js+j, d_ks+k, v) = bf_src.prim(s_is+i, s_js+j, s_ks+k, v);
                    }
                }
            }
        }
    }
}

// ===========================================================================
// 2. 接口残差平均 (Average Interface Residuals)
// [修正版] 增加了 n_cells <= 0 的检查，防止 vector::reserve 报错
// ===========================================================================
void HaloExchanger::average_interface_residuals(orion::bc::BCData& bc, orion::preprocess::FlowFieldSet& fs) {
    int myrank = orion::core::Runtime::rank();
    int ng = fs.ng; 

    // 使用临时结构存储本地拷贝的数据，防止读写竞争
    struct LocalData {
        CommTask task;
        std::vector<double> buffer; // 存储 dq/vol
    };
    std::vector<LocalData> local_data_list;

    reqs_.clear();
    send_buffers_.clear();
    recv_buffers_.clear();

    std::vector<CommTask> send_tasks;
    std::vector<CommTask> recv_tasks;

    auto GetFaceIndices = [&](const CommTask& task, int& is, int& ie, int& js, int& je, int& ks, int& ke) {
        // 直接使用 bc 定义的范围 (1-based)，转换为 0-based 并加上 ghost 偏移 ng
        is = task.s_st[0] - 1 + ng;
        ie = task.s_ed[0] - 1 + ng;
        js = task.s_st[1] - 1 + ng;
        je = task.s_ed[1] - 1 + ng;
        ks = task.s_st[2] - 1 + ng;
        ke = task.s_ed[2] - 1 + ng;
    };

    // 1. Identify & Pack Tasks
    for (int nb : fs.local_block_ids) {
        const auto& bcb = bc.block_bc[nb];
        auto& bf = fs.blocks[nb];

        for (int nr = 0; nr < bcb.nregions; ++nr) {
            const auto& reg = bcb.regions[nr];
            if (reg.bctype < 0) { // Interface
                int target_nb = reg.nbt - 1;
                int target_rank = bc.block_pid[target_nb] - 1;

                CommTask task;
                task.global_nb = nb;
                task.target_nb = target_nb;
                task.target_rank = target_rank;
                task.target_nr = reg.ibcwin; // 1-based target region index
                task.nr = nr; // local region index
                task.s_st = reg.s_st;
                task.s_ed = reg.s_ed;

                // 提取数据: DQ / Vol
                int is, ie, js, je, ks, ke;
                GetFaceIndices(task, is, ie, js, je, ks, ke);
                
                long long n_cells = (long long)(ie - is + 1) * (je - js + 1) * (ke - ks + 1);
                
                // [关键修正] 检查退化面或无效范围，防止 reserve 崩溃
                if (n_cells <= 0) continue;

                std::vector<double> buf;
                buf.reserve(n_cells * 5);

                for (int k = ks; k <= ke; ++k) {
                    for (int j = js; j <= je; ++j) {
                        for (int i = is; i <= ie; ++i) {
                            double vol = bf.vol(i, j, k);
                            double vol_inv = (vol > 1e-30) ? 1.0/vol : 0.0;
                            for(int m=0; m<5; ++m) {
                                buf.push_back(bf.dq(i, j, k, m) * vol_inv);
                            }
                        }
                    }
                }

                if (target_rank == myrank) {
                    // 本地拷贝：先缓存起来，等所有数据准备好后再 Update
                    LocalData ld;
                    ld.task = task;
                    ld.buffer = std::move(buf);
                    local_data_list.push_back(std::move(ld));
                } else {
                    send_tasks.push_back(task);
                    recv_tasks.push_back(task);
                    send_buffers_.push_back(std::move(buf));
                }
            }
        }
    }

    auto GenerateTag = [](int src_nb, int src_nr_1based) {
        return src_nb * 1000 + src_nr_1based;
    };

    // 2. MPI Isend
    for (size_t i = 0; i < send_tasks.size(); ++i) {
        const auto& task = send_tasks[i];
        int tag = GenerateTag(task.global_nb, task.nr + 1);
        
        MPI_Request req;
        MPI_Isend(send_buffers_[i].data(), (int)send_buffers_[i].size(), MPI_DOUBLE, 
                  task.target_rank, tag, MPI_COMM_WORLD, &req);
        reqs_.push_back(req);
    }

    // 3. MPI Irecv
    for (const auto& task : recv_tasks) {
        int tag = GenerateTag(task.target_nb, task.target_nr);
        
        int is, ie, js, je, ks, ke;
        GetFaceIndices(task, is, ie, js, je, ks, ke);
        long long n_cells = (long long)(ie - is + 1) * (je - js + 1) * (ke - ks + 1);
        
        // 如果 n_cells <= 0, 在上面循环中已经 skip 了，这里也应该 skip
        if (n_cells <= 0) continue;

        std::vector<double> buf(n_cells * 5);
        recv_buffers_.push_back(std::move(buf)); // Extend vector lifetime
        
        MPI_Request req;
        MPI_Irecv(recv_buffers_.back().data(), (int)recv_buffers_.back().size(), MPI_DOUBLE, 
                  task.target_rank, tag, MPI_COMM_WORLD, &req);
        reqs_.push_back(req);
    }

    // 4. Wait MPI
    if (!reqs_.empty()) {
        MPI_Waitall((int)reqs_.size(), reqs_.data(), MPI_STATUSES_IGNORE);
    }

    // 5. Process MPI Recv (Update)
    for (size_t i = 0; i < recv_tasks.size(); ++i) {
        const auto& task = recv_tasks[i];
        auto& bf = fs.blocks[task.global_nb];
        const double* ptr = recv_buffers_[i].data();

        int is, ie, js, je, ks, ke;
        GetFaceIndices(task, is, ie, js, je, ks, ke);

        int idx = 0;
        for (int k = ks; k <= ke; ++k) {
            for (int j = js; j <= je; ++j) {
                for (int i = is; i <= ie; ++i) {
                    double vol = bf.vol(i, j, k);
                    for(int m=0; m<5; ++m) {
                        double dq_neigh_per_vol = ptr[idx++];
                        // Average: 0.5 * (dq_self + dq_neigh_per_vol * vol_self)
                        bf.dq(i, j, k, m) = 0.5 * (bf.dq(i, j, k, m) + dq_neigh_per_vol * vol);
                    }
                }
            }
        }
    }

    // 6. Process Local Copy (Update)
    for (const auto& ld : local_data_list) {
        const auto& task = ld.task;
        
        // 目标块
        auto& bf_dst = fs.blocks[task.target_nb];
        
        // 目标区域信息
        int t_nr = task.target_nr - 1;
        const auto& bcb_dst = bc.block_bc[task.target_nb];
        const auto& reg_dst = bcb_dst.regions[t_nr];
        
        // 目标索引
        CommTask task_dst_view;
        task_dst_view.s_st = reg_dst.s_st;
        task_dst_view.s_ed = reg_dst.s_ed;
        
        int is, ie, js, je, ks, ke;
        GetFaceIndices(task_dst_view, is, ie, js, je, ks, ke);
        
        const double* ptr = ld.buffer.data();
        int idx = 0;
        
        for (int k = ks; k <= ke; ++k) {
            for (int j = js; j <= je; ++j) {
                for (int i = is; i <= ie; ++i) {
                    double vol = bf_dst.vol(i, j, k);
                    for(int m=0; m<5; ++m) {
                        double dq_neigh_per_vol = ptr[idx++]; // From Source
                        bf_dst.dq(i, j, k, m) = 0.5 * (bf_dst.dq(i, j, k, m) + dq_neigh_per_vol * vol);
                    }
                }
            }
        }
    }
}

} // namespace orion::solver