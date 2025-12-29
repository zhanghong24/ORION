#include "solver/ResidualMonitor.hpp"
#include "core/Runtime.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <mpi.h>

namespace orion::solver {

ResidualStats ResidualMonitor::compute(const orion::preprocess::FlowFieldSet& fs, 
                                       const orion::core::Params& params)
{
    ResidualStats stats;
    
    double sum_sq_res = 0.0;
    long long total_points = 0;
    
    double local_max_res = -1.0;
    int local_max_loc[5] = {-1, -1, -1, -1, -1}; // local_nb_idx, i, j, k, m

    int ng = fs.ng;
    int method = params.control.method;
    // Fortran: m1 = 1 + method. Loops from m1 to n-1.
    // If method=1 (Finite Diff), m1=2. Loop 2..ni-1.
    // In C++ (0-based):
    // 1-based "2" -> 0-based "1".
    // 1-based "ni-1" -> 0-based "ni-2".
    // So loop range: [1, ni-2]. (Internal points, excluding boundary cells)
    // ng usually >= 2.
    // Let's use internal domain range.
    
    // Fortran logic:
    // dtime1 = 1.0 / (dtdt * vol)
    // dresm = abs(dq) * dtime1
    // res += dresm^2
    
    for (int nb_idx = 0; nb_idx < (int)fs.local_block_ids.size(); ++nb_idx) {
        int nb = fs.local_block_ids[nb_idx];
        const auto& bf = fs.blocks[nb];
        
        const auto& dims = bf.dq.dims();
        int nx = dims[0], ny = dims[1], nz = dims[2];
        
        // Loop range: internal cells
        // Fortran: m1 .. n-1.
        // Assuming method=1 => 2 .. n-1 (1-based) => 1 .. n-2 (0-based)
        // Adjust for ghost ng.
        // Start: ng. End: nx - ng.
        // Fortran loop seems to skip first and last internal point?
        // "do i=m1, ni-1". if ni=10, m1=2. i=2..9.
        // C++ indices with ng:
        // Physical domain starts at ng.
        // Let's iterate all valid internal cells where dq is updated.
        // Usually ng to nx-ng.
        
        int ist = ng, ied = nx - ng;
        int jst = ng, jed = ny - ng;
        int kst = ng, ked = nz - ng;

        for (int k = kst; k < ked; ++k) {
            for (int j = jst; j < jed; ++j) {
                for (int i = ist; i < ied; ++i) {
                    
                    double vol = bf.vol(i, j, k);
                    double dt = bf.dt(i, j, k);
                    
                    // Prevent div by zero
                    if (vol < 1e-30) vol = 1e-30;
                    if (dt < 1e-30) dt = 1e-30;
                    
                    double inv_factor = 1.0 / (dt * vol);

                    for (int m = 0; m < 5; ++m) {
                        double dq_val = std::abs(bf.dq(i, j, k, m));
                        double res_val = dq_val * inv_factor;
                        
                        // L2 Accumulation
                        sum_sq_res += res_val * res_val;
                        
                        // Max Check
                        if (res_val > local_max_res) {
                            local_max_res = res_val;
                            local_max_loc[0] = nb; // Global Block ID
                            local_max_loc[1] = i - ng + 1; // Convert to 1-based physical index
                            local_max_loc[2] = j - ng + 1;
                            local_max_loc[3] = k - ng + 1;
                            local_max_loc[4] = m + 1;
                        }
                    }
                    total_points += 5; // 5 vars per point
                }
            }
        }
    }

    // --- MPI Reduction ---
    double global_sum_sq = 0.0;
    long long global_points = 0;
    
    MPI_Allreduce(&sum_sq_res, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&total_points, &global_points, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    // Find global max
    struct {
        double val;
        int rank;
    } local_max, global_max;
    
    local_max.val = local_max_res;
    local_max.rank = orion::core::Runtime::rank();
    
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    
    stats.max_residual = global_max.val;
    
    // Broadcast location from the rank that has max
    if (global_max.rank == local_max.rank) {
        // If I am the max holder, I send my location
        // But wait, MPI_Bcast needs root.
        // We can just use a struct broadcast or simply rely on standard output from root?
        // Actually, only root needs to know for printing.
        // But if we want to return stats to everyone...
        std::copy(std::begin(local_max_loc), std::end(local_max_loc), std::begin(stats.max_loc));
    }
    
    // Broadcast the location array from the winner rank
    MPI_Bcast(stats.max_loc, 5, MPI_INT, global_max.rank, MPI_COMM_WORLD);
    
    sum_sq_res = global_sum_sq;
    total_points = global_points;

    if (total_points > 0) {
        stats.rms_residual = std::sqrt(sum_sq_res / total_points);
    }

    return stats;
}

} // namespace orion::solver