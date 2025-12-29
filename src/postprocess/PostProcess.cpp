#include "postprocess/PostProcess.hpp"
#include "core/Runtime.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <filesystem>
#include <vector>
#include <cstring>
#include <mpi.h> // [修改] 强制包含 MPI 头文件

namespace fs = std::filesystem;

namespace orion::postprocess {

// ---------------------------------------------------------------------------
// 辅助类：用于管理二进制偏移量
// ---------------------------------------------------------------------------
struct OffsetManager {
    using HeaderType = uint32_t; 
    size_t current_offset = 0;

    size_t next_block(size_t num_elements, size_t element_size) {
        size_t start = current_offset;
        current_offset += sizeof(HeaderType) + num_elements * element_size;
        return start;
    }
};

// 辅助函数：写入二进制数据块
template <typename T>
static void write_binary_array(std::ofstream& out, const std::vector<T>& data) {
    uint32_t bytes = static_cast<uint32_t>(data.size() * sizeof(T));
    out.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()), bytes);
    }
}

// ---------------------------------------------------------------------------
// 主入口
// ---------------------------------------------------------------------------
void PostProcess::write_solution(const orion::preprocess::FlowFieldSet& fs, 
                                 const orion::mesh::MultiBlockGrid& grid,
                                 const orion::core::Params& params,
                                 int step)
{
    int my_rank = orion::core::Runtime::rank();
    int ng = fs.ng;

    // 1. 路径处理
    std::string path_prefix = params.filename.flowname;
    if (path_prefix.empty()) path_prefix = "output/flow";

    std::filesystem::path p(path_prefix);
    std::string output_dir = p.parent_path().string();
    std::string file_stem = p.filename().string();

    if (output_dir.empty()) output_dir = ".";

    if (my_rank == 0) {
        if (!std::filesystem::exists(output_dir)) {
            try {
                std::filesystem::create_directories(output_dir);
            } catch (const std::exception& e) {
                std::cerr << "[PostProcess] Error creating directory: " << e.what() << std::endl;
            }
        }
    }
    
    // [修改] 强制同步，确保目录已创建
    MPI_Barrier(MPI_COMM_WORLD);

    // 2. 输出本地块 (Binary VTS)
    for (int nb : fs.local_block_ids) {
        std::ostringstream oss;
        oss << output_dir << "/" << file_stem << "_" 
            << std::setfill('0') << std::setw(6) << step 
            << "_block_" << nb << ".vts";
        
        const auto& bg = grid.blocks[nb];
        const auto& bf = fs.blocks.at(nb);
        
        write_block_vts_binary(nb, bf, bg, oss.str(), params, ng);
    }

    // 3. 输出 VTM 索引 (Rank 0 only)
    
    // 获取本地最大 block id
    int max_id_local = -1;
    if (!fs.local_block_ids.empty()) {
        max_id_local = fs.local_block_ids.back();
    }
    
    int max_id_global = 0;
    
    // [修改] 强制执行全局规约，不再依赖宏定义
    MPI_Reduce(&max_id_local, &max_id_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        // Block ID 是从 0 开始的，所以总数是 max_id + 1
        int total_blocks = max_id_global + 1;
        write_global_vtm(file_stem, total_blocks, step, output_dir);
    }
}

// ---------------------------------------------------------------------------
// 单块输出核心 (Binary)
// ---------------------------------------------------------------------------
void PostProcess::write_block_vts_binary(int block_id, 
                                         const orion::preprocess::BlockField& bf, 
                                         const orion::mesh::BlockGrid& bg, 
                                         const std::string& filepath,
                                         const orion::core::Params& params,
                                         int ng)
{
    std::ofstream out(filepath, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[PostProcess] Failed to open file: " << filepath << std::endl;
        return;
    }

    int nx = bg.idim;
    int ny = bg.jdim;
    int nz = bg.kdim;
    size_t num_points = (size_t)nx * ny * nz;

    if (num_points == 0) return;

    int ext_i = nx - 1;
    int ext_j = ny - 1;
    int ext_k = nz - 1;

    // 2. 计算偏移量
    OffsetManager om;
    size_t off_pts = om.next_block(num_points * 3, sizeof(float)); 
    size_t off_rho = om.next_block(num_points, sizeof(float));     
    size_t off_p   = om.next_block(num_points, sizeof(float));     
    size_t off_T   = om.next_block(num_points, sizeof(float));     
    size_t off_Ma  = om.next_block(num_points, sizeof(float));     
    size_t off_Vel = om.next_block(num_points * 3, sizeof(float)); 

    // 3. 写入 XML Header
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
    out << "  <StructuredGrid WholeExtent=\"0 " << ext_i << " 0 " << ext_j << " 0 " << ext_k << "\">\n";
    out << "    <Piece Extent=\"0 " << ext_i << " 0 " << ext_j << " 0 " << ext_k << "\">\n";

    // 3.1 Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_pts << "\"/>\n";
    out << "      </Points>\n";

    // 3.2 PointData
    out << "      <PointData Scalars=\"Density\" Vectors=\"Velocity\">\n";
    out << "        <DataArray type=\"Float32\" Name=\"Density\" format=\"appended\" offset=\"" << off_rho << "\"/>\n";
    out << "        <DataArray type=\"Float32\" Name=\"Pressure\" format=\"appended\" offset=\"" << off_p << "\"/>\n";
    out << "        <DataArray type=\"Float32\" Name=\"Temperature\" format=\"appended\" offset=\"" << off_T << "\"/>\n";
    out << "        <DataArray type=\"Float32\" Name=\"Mach\" format=\"appended\" offset=\"" << off_Ma << "\"/>\n";
    out << "        <DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << off_Vel << "\"/>\n";
    out << "      </PointData>\n";

    out << "    </Piece>\n";
    out << "  </StructuredGrid>\n";

    // 4. 写入二进制数据
    out << "  <AppendedData encoding=\"raw\">\n";
    out << "    _"; 

    std::vector<float> buffer_scalar; 
    buffer_scalar.reserve(num_points);
    std::vector<float> buffer_vector; 
    buffer_vector.reserve(num_points * 3);

    // --- A. Points ---
    buffer_vector.clear();
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                buffer_vector.push_back((float)bg.x(i,j,k));
                buffer_vector.push_back((float)bg.y(i,j,k));
                buffer_vector.push_back((float)bg.z(i,j,k));
            }
        }
    }
    write_binary_array(out, buffer_vector);

    // --- Helper for Scalars ---
    auto extract_scalar = [&](auto func) {
        buffer_scalar.clear();
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    buffer_scalar.push_back((float)func(i + ng, j + ng, k + ng));
                }
            }
        }
        write_binary_array(out, buffer_scalar);
    };

    // --- B. Density ---
    extract_scalar([&](int ix, int iy, int iz) { return bf.prim(ix, iy, iz, 0); });

    // --- C. Pressure ---
    extract_scalar([&](int ix, int iy, int iz) { return bf.prim(ix, iy, iz, 4); });

    // --- D. Temperature ---
    extract_scalar([&](int ix, int iy, int iz) { 
        double r = bf.prim(ix, iy, iz, 0);
        double p = bf.prim(ix, iy, iz, 4);
        return (std::abs(r) > 1e-30) ? (p / r) : 0.0;
    });

    // --- E. Mach ---
    double gamma = params.inflow.gama;
    extract_scalar([&](int ix, int iy, int iz) {
        double r = bf.prim(ix, iy, iz, 0);
        double u = bf.prim(ix, iy, iz, 1);
        double v = bf.prim(ix, iy, iz, 2);
        double w = bf.prim(ix, iy, iz, 3);
        double p = bf.prim(ix, iy, iz, 4);
        double v2 = u*u + v*v + w*w;
        double c2 = gamma * p / r;
        return (c2 > 1e-30) ? std::sqrt(v2 / c2) : 0.0;
    });

    // --- F. Velocity ---
    buffer_vector.clear();
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int ix = i + ng, iy = j + ng, iz = k + ng;
                buffer_vector.push_back((float)bf.prim(ix, iy, iz, 1));
                buffer_vector.push_back((float)bf.prim(ix, iy, iz, 2));
                buffer_vector.push_back((float)bf.prim(ix, iy, iz, 3));
            }
        }
    }
    write_binary_array(out, buffer_vector);

    out << "\n  </AppendedData>\n";
    out << "</VTKFile>\n";
    out.close();
}

void PostProcess::write_global_vtm(const std::string& base_filename,
                                   int total_blocks,
                                   int step,
                                   const std::string& output_dir)
{
    std::ostringstream filename_ss;
    filename_ss << output_dir << "/" << base_filename << "_" 
                << std::setfill('0') << std::setw(6) << step << ".vtm";
    
    std::ofstream out(filename_ss.str());
    if (!out.is_open()) return;

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out << "  <vtkMultiBlockDataSet>\n";

    for (int nb = 0; nb < total_blocks; ++nb) {
        std::ostringstream vts_name;
        vts_name << base_filename << "_" 
                 << std::setfill('0') << std::setw(6) << step 
                 << "_block_" << nb << ".vts";
        
        out << "    <DataSet index=\"" << nb << "\" file=\"" << vts_name.str() << "\"/>\n";
    }

    out << "  </vtkMultiBlockDataSet>\n";
    out << "</VTKFile>\n";
    out.close();
    
    std::cout << "[PostProcess] Wrote step " << step << " to " << filename_ss.str() << std::endl;
}

} // namespace orion::postprocess