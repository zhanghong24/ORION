#pragma once
#include "bc/BCData.hpp"

namespace orion::bc {

/// Build bcindexs for each block, same behavior as Fortran set_bc_index.
/// bcindexs stores 1-based region indices ordered by (-1,4,5,6,2,3).
void set_bc_index(BCData& bc);

} // namespace orion::bc
