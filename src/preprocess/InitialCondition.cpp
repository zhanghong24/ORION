#include "preprocess/InitialCondition.hpp"
#include "core/Runtime.hpp"
#include <iostream>

namespace orion::preprocess {

void initialize_flow(const orion::core::Params& params, FlowFieldSet& fs) {
    
    // 1. 检查启动模式 (使用 params.control.nstart)
    if (params.control.nstart != 0) {
        // Restart 模式暂时留空 (对应 Fortran: call init_nstart1)
        if (orion::core::Runtime::is_root()) {
            std::cout << "[Init] Restart mode (nstart=" << params.control.nstart 
                      << ") detected but logic is NOT implemented yet.\n";
            std::cout << "       Skipping initialization.\n";
        }
        return;
    }

    // --- Cold Start (nstart == 0) ---
    // 对应 Fortran: call init_nstart0

    if (orion::core::Runtime::is_root()) {
        std::cout << "[Init] Performing Cold Start (nstart=0)...\n";
    }

    const auto& F = params.inflow;

    // 2. 遍历本进程所有的 Block
    for (int nb_idx : fs.local_block_ids) {
        auto& bf = fs.blocks[nb_idx];
        
        // 确保数组已分配
        if (!bf.allocated()) {
            throw std::runtime_error("FlowField not allocated in initialize_flow");
        }

        // 获取数组维度 (包含 Ghost Layers)
        const auto& dims = bf.q.dims();
        int nx = dims[0];
        int ny = dims[1];
        int nz = dims[2];

        // 3. 填充初值
        // Fortran: r=roo, u=uoo, v=voo, w=woo, p=poo, t=1.0
        // C++: prim = [rho, u, v, w, p, T]
        //      q    = [rho, rho*u, rho*v, rho*w, E]
        
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    
                    // --- 设置守恒变量 Q (Conservative Variables) ---
                    // 使用 init_inflow 中计算好的 q1oo ~ q5oo
                    bf.q(i, j, k, 0) = F.q1oo; // rho
                    bf.q(i, j, k, 1) = F.q2oo; // rho * u
                    bf.q(i, j, k, 2) = F.q3oo; // rho * v
                    bf.q(i, j, k, 3) = F.q4oo; // rho * w
                    bf.q(i, j, k, 4) = F.q5oo; // E

                    // --- 设置原始变量 Prim (Primitive Variables) ---
                    // 假设 prim 布局: 0:rho, 1:u, 2:v, 3:w, 4:p, 5:T
                    if (fs.nprim >= 5) {
                        bf.prim(i, j, k, 0) = F.roo; // rho
                        bf.prim(i, j, k, 1) = F.uoo; // u
                        bf.prim(i, j, k, 2) = F.voo; // v
                        bf.prim(i, j, k, 3) = F.woo; // w
                        bf.prim(i, j, k, 4) = F.poo; // p
                    }
                    if (fs.nprim >= 6) {
                        bf.prim(i, j, k, 5) = F.too; // T (即 1.0)
                    }

                    // --- 初始化 dq 为 0 ---
                    for (int m = 0; m < fs.nvar; ++m) {
                        bf.dq(i, j, k, m) = 0.0;
                    }
                }
            }
        }
    }

    if (orion::core::Runtime::is_root()) {
        std::cout << "[Init] Flow field initialized with freestream conditions.\n";
    }
}

// =======================================================================
// 更新温度场 (From EOS)
// =======================================================================
void update_temperature(const orion::core::Params& params, FlowFieldSet& fs) {
    // 1. 检查粘性开关 (params.flowtype.nvis)
    // Fortran: if(nvis == 1) then call get_t_ini
    if (params.flowtype.nvis == 0) {
        return; // 无粘计算通常不需要显式更新 T (或者已经由能量方程隐含)
    }

    if (orion::core::Runtime::is_root()) {
        std::cout << "[Init] Updating Temperature field from EOS (nvis=" 
                  << params.flowtype.nvis << ")...\n";
    }

    const double gama = params.inflow.gama;
    const double mach = params.inflow.moo;
    const double mach2 = mach * mach;

    // 2. 遍历所有块
    for (int nb_idx : fs.local_block_ids) {
        auto& bf = fs.blocks[nb_idx];
        
        // 确保 prim 至少有 6 个分量 (rho,u,v,w,p,T)
        if (fs.nprim < 6) {
            // 如果 nprim 不够存温度，跳过并警告
            static bool warned = false;
            if (!warned) {
                std::cerr << "[Warning] nprim < 6, cannot store Temperature!\n";
                warned = true;
            }
            continue;
        }

        const auto& dims = bf.prim.dims();
        int nx = dims[0];
        int ny = dims[1];
        int nz = dims[2];

        // 3. 遍历网格点更新 T
        // 公式: T = M^2 * (gamma * p / rho)
        
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    double rho = bf.prim(i, j, k, 0);
                    double p   = bf.prim(i, j, k, 4);
                    
                    // 简单的除零保护
                    if (rho < 1.0e-30) rho = 1.0e-30;

                    double a2 = gama * p / rho; // a^2 (声速平方)
                    double t_val = mach2 * a2;  // T

                    bf.prim(i, j, k, 5) = t_val;
                }
            }
        }
    }
}

} // namespace orion::preprocess