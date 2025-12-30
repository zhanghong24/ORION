#include "bc/BCPreprocess.hpp"
#include <array>
#include <stdexcept>
#include <sstream>

namespace orion::bc {

// Helper to flatten 3D index
static int idx3(int i, int j, int k, int dim_i, int dim_j) {
    return (k * dim_j + j) * dim_i + i;
}

// ===========================================================================
// Replicating Fortran subroutine analyze_bc_connect
// ===========================================================================
static void analyze_bc_connect(BCRegion& reg) {
    using std::abs;
    
    // Fortran indices are 1-based. We keep calculations in 1-based logic 
    // to match Fortran strictly, then convert to 0-based for storage if needed,
    // or just store results relative to the start.
    // Here we strictly follow Fortran logic.

    int s_st[3] = {reg.s_st[0], reg.s_st[1], reg.s_st[2]};
    int s_ed[3] = {reg.s_ed[0], reg.s_ed[1], reg.s_ed[2]};
    int t_st[3] = {reg.t_st[0], reg.t_st[1], reg.t_st[2]};
    int t_ed[3] = {reg.t_ed[0], reg.t_ed[1], reg.t_ed[2]};
    
    int s_nd = reg.s_nd; // 1-based: 1,2,3

    // 1. Determine Target Orientation (t_nd, t_lr)
    for (int m = 0; m < 3; ++m) {
        reg.t_lr3d[m] = 0;
        if (t_st[m] == t_ed[m]) {
            reg.t_nd = m + 1; // 1-based
            if (t_st[m] == 1) {
                reg.t_lr = -1;
                reg.t_lr3d[m] = -1;
            } else {
                reg.t_lr = 1;
                reg.t_lr3d[m] = 1;
            }
            reg.t_fix = t_st[m];
        }
    }
    int t_nd = reg.t_nd;

    // 2. Determine Orientation Matrix (s_t_direction)
    int s_t_dirction[3][3] = {{0}}; // [n][m] -> t[n] vs s[m]

    for (int m = 0; m < 3; ++m) {     // s index (0-2)
        for (int n = 0; n < 3; ++n) { // t index (0-2)
            if ( (m + 1) != s_nd && (n + 1) != t_nd ) {
                int js1 = s_st[m];
                if (abs(js1) < abs(s_ed[m])) js1 = s_ed[m];

                int js2 = t_st[n];
                if (abs(js2) < abs(t_ed[n])) js2 = t_ed[n];

                if (js1 * js2 > 0) {
                    s_t_dirction[n][m] = 1;
                    // The other diagonal logic from Fortran:
                    // s_t_dirction(t_nd, s_nd) = 1
                    s_t_dirction[t_nd-1][s_nd-1] = 1; 
                    // s_t_dirction(6-n-t_nd, 6-m-s_nd) = 1 (using 1-based indices math)
                    // (n+1) + t_nd + unknown = 6 => unknown = 6 - (n+1) - t_nd
                    int rem_t = 6 - (n+1) - t_nd; 
                    int rem_s = 6 - (m+1) - s_nd;
                    s_t_dirction[rem_t-1][rem_s-1] = 1;
                    goto label_10;
                }
            }
        }
    }
label_10:;

    // 3. Determine Signs (s_sign, t_sign, st_sign)
    int s_sign[3], t_sign[3], st_sign[3];
    int abs_s_st[3], abs_s_ed[3]; // Working vars

    for (int m = 0; m < 3; ++m) {
        s_sign[m] = 1;
        abs_s_st[m] = abs(s_st[m]);
        abs_s_ed[m] = abs(s_ed[m]);
        
        if ((m + 1) != s_nd) {
            if (abs_s_st[m] > abs_s_ed[m]) s_sign[m] = -1;
        }

        t_sign[m] = 1;
        int abs_t_st = abs(t_st[m]);
        int abs_t_ed = abs(t_ed[m]);
        if ((m + 1) != t_nd) {
            if (abs_t_st > abs_t_ed) t_sign[m] = -1;
        }
    }

    for (int m = 0; m < 3; ++m) {
        int co = 0; // index in t (0-2)
        for (int n = 0; n < 3; ++n) {
            if (s_t_dirction[n][m] == 1) co = n;
        }
        st_sign[m] = t_sign[co] * s_sign[m];
    }

    // 4. Allocate and Fill Mapping Arrays
    // Compute dimensions of the Local Ghost Window (Source range in logic)
    // Note: Fortran allocates image(js1:js2, ...). We use 0-based vector.
    // We normalize loops to 0..dim-1.
    
    int js1 = abs(s_st[0]), js2 = abs(s_ed[0]);
    int ks1 = abs(s_st[1]), ks2 = abs(s_ed[1]);
    int ls1 = abs(s_st[2]), ls2 = abs(s_ed[2]);

    // Ensure min/max for size calculation
    int i_min = std::min(js1, js2), i_max = std::max(js1, js2);
    int j_min = std::min(ks1, ks2), j_max = std::max(ks1, ks2);
    int k_min = std::min(ls1, ls2), k_max = std::max(ls1, ls2);

    reg.map_dims[0] = i_max - i_min + 1;
    reg.map_dims[1] = j_max - j_min + 1;
    reg.map_dims[2] = k_max - k_min + 1;
    
    int total_points = reg.map_dims[0] * reg.map_dims[1] * reg.map_dims[2];
    reg.image.resize(total_points);
    reg.jmage.resize(total_points);
    reg.kmage.resize(total_points);

    // Loop over Local Ghost Cells (s_st -> s_ed)
    // We track 'idx' for the flat vectors.
    
    // Fortran loop: do i = s_st(1), s_ed(1), s_sign(1)
    // We simulate this manually.
    
    int count = 0;
    // Normalized loops for buffer filling order (usually k-j-i in memory if contiguous?)
    // Wait, Fortran allocates (js1:js2, ks1:ks2...). 
    // Access is image(i,j,k). 
    // To enable image[idx3(i-imin, ...)] lookup, we fill accordingly.

    // Loop variables for calculating values
    int i_curr = abs(s_st[0]);
    int j_curr_start = abs(s_st[1]);
    int k_curr_start = abs(s_st[2]);

    // We iterate 0..dim to fill the vector sequentially
    // But the VALUE depends on i_curr, which steps by s_sign
    for (int k_idx = 0; k_idx < reg.map_dims[2]; ++k_idx) {
        int k = (s_sign[2] == 1) ? (k_min + k_idx) : (k_max - k_idx); // Fortran logic simulation?
        // Actually, Fortran loop `do k = s_st, s_ed, s_sign`.
        // If s_st < s_ed, s_sign=1. Loop min to max.
        // If s_st > s_ed, s_sign=-1. Loop max to min.
        // Let's stick to the Fortran loop structure exactly.
    }

    // Rewrite loops to match Fortran exactly, but fill flat vector
    // We need to map (i,j,k) to flat index 0..N-1.
    // Standard convention: (i-imin) + (j-jmin)*dim_i + ...
    
    for (int k = abs(s_st[2]); k != abs(s_ed[2]) + s_sign[2]; k += s_sign[2]) {
        for (int j = abs(s_st[1]); j != abs(s_ed[1]) + s_sign[1]; j += s_sign[1]) {
            for (int i = abs(s_st[0]); i != abs(s_ed[0]) + s_sign[0]; i += s_sign[0]) {
                
                int idelt = (i - abs(s_st[0])) * st_sign[0];
                int jdelt = (j - abs(s_st[1])) * st_sign[1];
                int kdelt = (k - abs(s_st[2])) * st_sign[2];

                // Calculate target coordinates (1-based result)
                // s_t_dirction indices are 0-based in C++, so [0][0] matches (1,1)
                
                int co_i = s_t_dirction[0][0]*idelt + s_t_dirction[0][1]*jdelt + s_t_dirction[0][2]*kdelt;
                int val_i = abs(t_st[0]) + co_i;

                int co_j = s_t_dirction[1][0]*idelt + s_t_dirction[1][1]*jdelt + s_t_dirction[1][2]*kdelt;
                int val_j = abs(t_st[1]) + co_j;

                int co_k = s_t_dirction[2][0]*idelt + s_t_dirction[2][1]*jdelt + s_t_dirction[2][2]*kdelt;
                int val_k = abs(t_st[2]) + co_k;

                // Store in vector at position corresponding to (i,j,k)
                int local_i = i - i_min;
                int local_j = j - j_min;
                int local_k = k - k_min;
                int flat_idx = idx3(local_i, local_j, local_k, reg.map_dims[0], reg.map_dims[1]);

                reg.image[flat_idx] = val_i;
                reg.jmage[flat_idx] = val_j;
                reg.kmage[flat_idx] = val_k;
            }
        }
    }
}

void set_bc_index(BCData& bc)
{
  constexpr std::array<int, 6> bclists{{-1, 4, 5, 6, 2, 3}};

  const int nblocks = bc.number_of_blocks;
  if (nblocks < 0) {
    throw std::runtime_error("set_bc_index: invalid number_of_blocks < 0");
  }
  if ((int)bc.block_bc.size() != nblocks) {
    throw std::runtime_error("set_bc_index: bc.block_bc size mismatch with number_of_blocks");
  }

  for (int nb = 0; nb < nblocks; ++nb) {
    auto& blk = bc.block_bc[(std::size_t)nb];

    const int nrmax = (int)blk.regions.size();
    blk.nregions = nrmax; // keep consistent if you store it

    blk.bcindexs.clear();
    blk.bcindexs.reserve((std::size_t)nrmax);

    int i = 0;
    for (int idbc : bclists) {
      for (int nr = 0; nr < nrmax; ++nr) {
        const int bctype = blk.regions[(std::size_t)nr].bctype;
        if (bctype == idbc) {
          ++i;
          // Fortran stores 1-based index: nr (1..nrmax)
          blk.bcindexs.push_back(nr + 1);
        }
      }
    }

    if (i != nrmax) {
      // 找出哪些 bctype 不在 bclists 里，方便定位
      auto in_list = [&](int t) {
        for (int v : bclists) if (v == t) return true;
        return false;
      };
      std::ostringstream oss;
      oss << "ERROR: incorrect BC types in block " << (nb + 1)
          << " (name=" << blk.blockname << "). "
          << "Expected types among {-1,4,5,6,2,3}. Found: ";
      for (int nr = 0; nr < nrmax; ++nr) {
        int t = blk.regions[(std::size_t)nr].bctype;
        if (!in_list(t)) {
          oss << "[region " << (nr + 1) << " type " << t << "] ";
        }
      }
      throw std::runtime_error(oss.str());
    }
  }
}

void prepare_bc_topology(BCData& bc) {
    for (auto& bcb : bc.block_bc) {
        for (auto& reg : bcb.regions) {
            // Fortran Logic:
            // 1. Normalize s_st/s_ed to positive and set s_lr3d (already done or needs doing)
            // 2. If bctype < 0, call analyze_bc_connect
            
            // Helper: setup s_lr3d (Partial replication of analyze_bc lines 967-970)
            for(int m=0; m<3; ++m) {
                reg.s_lr3d[m] = 0;
                if( abs(reg.s_st[m]) == abs(reg.s_ed[m]) ) {
                    // This is the face normal direction
                    if( abs(reg.s_st[m]) == 1 ) reg.s_lr3d[m] = -1; // Min face
                    else                        reg.s_lr3d[m] =  1; // Max face
                }
            }
            
            if (reg.is_connect()) {
                 analyze_bc_connect(reg);
            }
        }
    }
}

} // namespace orion::bc
