#include "bc/BCPreprocess.hpp"
#include <array>
#include <stdexcept>
#include <sstream>

namespace orion::bc {

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

} // namespace orion::bc
