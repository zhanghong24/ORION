// src/preprocess/DebugDump.hpp
#pragma once

#include "preprocess/FlowField.hpp"
#include "mesh/MultiBlockGrid.hpp"
#include "core/Runtime.hpp"

namespace orion::preprocess {

void debug_dump_local_blocks_and_bcflag(const mesh::MultiBlockGrid& grid,
                                        const FlowFieldSet& fs);

} // namespace orion::preprocess
