#pragma once
#include "preprocess/FlowField.hpp"
#include "mesh/MultiBlockGrid.hpp"
#include "core/Params.hpp"
#include <string>
#include <vector>

namespace orion::postprocess {

class PostProcess {
public:
    /**
     * @brief 输出流场结果到 VTK XML 二进制文件 (VTS/VTM)
     * 格式：Appended Raw Binary (Little Endian)
     */
    static void write_solution(const orion::preprocess::FlowFieldSet& fs, 
                               const orion::mesh::MultiBlockGrid& grid,
                               const orion::core::Params& params,
                               int step);

private:
    static void write_block_vts_binary(int block_id, 
                                       const orion::preprocess::BlockField& bf, 
                                       const orion::mesh::BlockGrid& bg, 
                                       const std::string& filepath,
                                       const orion::core::Params& params,
                                       int ng);

    static void write_global_vtm(const std::string& base_filename,
                                 int total_blocks,
                                 int step,
                                 const std::string& output_dir);
};

} // namespace orion::postprocess