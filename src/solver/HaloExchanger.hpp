#pragma once

#include "mesh/MultiBlockGrid.hpp"
#include "bc/BCData.hpp"
#include "preprocess/FlowField.hpp"
#include <vector>
#include <mpi.h>

namespace orion::solver {

class HaloExchanger {
public:
    HaloExchanger() = default;
    ~HaloExchanger() = default;

    /**
     * @brief 执行幽灵网格交换 (Primitive Variables: rho, u, v, w, p)
     * 对应 Fortran: exchange_bc
     */
    void exchange_bc(orion::bc::BCData& bc,
                     orion::preprocess::FlowFieldSet& fs);

    // [新增] 交换并平均接口残差 DQ (用于粘性通量计算后的平滑)
    // 对应 Fortran: communicate_dq_npp (exchange_bc_dq_vol + boundary_match_dq_2pm)
    void average_interface_residuals(bc::BCData& bc, orion::preprocess::FlowFieldSet& fs);

private:
    // MPI 请求句柄
    std::vector<MPI_Request> reqs_;
    
    // 发送/接收缓冲区 (避免临时对象析构)
    // 列表结构保证指针有效性
    std::vector<std::vector<double>> send_buffers_;
    std::vector<std::vector<double>> recv_buffers_;

    struct CommTask {
        int local_nb_idx; // 本地块在 fs.local_block_ids 中的下标
        int global_nb;    // 全局 Block ID
        int nr;           // 边界 Region 索引
        
        int target_nb;    // 目标 Block ID
        int target_rank;  // 目标 Rank
        int target_nr;    // 目标 Region 索引 (ibcwin)
        
        // 几何信息
        std::array<int,3> s_st;
        std::array<int,3> s_ed;
        int dir;    // 法向 (0,1,2)
        int inrout; // 方向 (1 or -1)
    };

    // 辅助函数：根据 Fortran 逻辑计算读写窗口
    void get_window(const CommTask& task, 
                    const orion::preprocess::BlockField& bf,
                    int ng, bool is_send,
                    int& i_start, int& i_end,
                    int& j_start, int& j_end,
                    int& k_start, int& k_end);

    // 内部辅助：处理本地块之间的直接拷贝 (无需 MPI)
    void process_local_copy_dq(bc::BCData& bc, orion::preprocess::FlowFieldSet& fs);

    // 内部辅助：MPI 通信逻辑
    void process_mpi_exchange_dq(bc::BCData& bc, orion::preprocess::FlowFieldSet& fs);
};

} // namespace orion::solver