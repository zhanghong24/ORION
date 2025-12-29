#include "bc/BCReader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <cctype>

namespace orion::bc {

static void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

static void consume_line(std::ifstream& fin) {
  fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

static std::string read_nonempty_line(std::ifstream& fin) {
  std::string line;
  while (std::getline(fin, line)) {
    auto l = line;
    while (!l.empty() && std::isspace((unsigned char)l.back())) l.pop_back();
    std::size_t p = 0;
    while (p < l.size() && std::isspace((unsigned char)l[p])) ++p;
    l = l.substr(p);
    if (!l.empty()) return l;
  }
  return {};
}

BCData BCReader::read_parallel_bc_root_only(const std::string& bcname,
                                            const orion::mesh::MultiBlockGrid& grid,
                                            int numprocs,
                                            const BCReaderOptions& opt)
{
  std::ifstream fin(bcname);
  require(fin.is_open(), "Failed to open bc file: " + bcname);

  BCData out;

  fin >> out.flow_solver_id;
  require(fin.good(), "Bad read: flow_solver_id");
  if (out.flow_solver_id != opt.mbsmarker) {
    throw std::runtime_error("BC file marker mismatch: got " +
      std::to_string(out.flow_solver_id) + ", expect " + std::to_string(opt.mbsmarker));
  }

  fin >> out.nprocs_in_file;
  require(fin.good(), "Bad read: nprocs in file");

  int ntms = -1;
  if (out.nprocs_in_file > 0 && (out.nprocs_in_file % numprocs) == 0) {
    ntms = out.nprocs_in_file / numprocs;
  }

  fin >> out.number_of_blocks;
  require(fin.good(), "Bad read: number_of_blocks");

  const int nblocks = grid.nblocks;
  require(out.number_of_blocks == nblocks,
          "BC blocks != grid blocks: bc=" + std::to_string(out.number_of_blocks) +
          " grid=" + std::to_string(nblocks));

  out.block_pid.resize(nblocks);
  out.block_bc.resize(nblocks);

  if (opt.verbose) {
    std::cout << "[BCReader] Read BC file: " << bcname << "\n";
    std::cout << "[BCReader] blocks: " << nblocks
              << ", file_nprocs=" << out.nprocs_in_file
              << ", mpi_nprocs=" << numprocs
              << ", ntms=" << ntms << "\n";
  }

  for (int nb0 = 0; nb0 < nblocks; ++nb0) {
    int pid = 0;
    fin >> pid;
    require(fin.good(), "Bad read: pid for block " + std::to_string(nb0+1));

    if (ntms > 0) {
      pid = (pid - 1) / ntms + 1;
    }
    out.block_pid[nb0] = pid;

    int imax=0, jmax=0, kmax=0;
    fin >> imax >> jmax >> kmax;
    require(fin.good(), "Bad read: dims for block " + std::to_string(nb0+1));

    const auto& gb = grid.blocks[nb0];
    int ndif = std::abs(imax - gb.idim) + std::abs(jmax - gb.jdim) + std::abs(kmax - gb.kdim);
    require(ndif == 0,
            "BC dims mismatch at block " + std::to_string(nb0+1) +
            " bc=(" + std::to_string(imax)+","+std::to_string(jmax)+","+std::to_string(kmax)+")" +
            " grid=(" + std::to_string(gb.idim)+","+std::to_string(gb.jdim)+","+std::to_string(gb.kdim)+")");

    consume_line(fin); 
    out.block_bc[nb0].blockname = read_nonempty_line(fin);
    require(!out.block_bc[nb0].blockname.empty(), "Missing blockname line");

    int nrmax = 0;
    fin >> nrmax;
    require(fin.good(), "Bad read: nrmax for block " + std::to_string(nb0+1));

    out.block_bc[nb0].nregions = nrmax;
    out.block_bc[nb0].regions.resize(nrmax);

    for (int r = 0; r < nrmax; ++r) {
      BCRegion reg;
      std::array<int,3> s_st{}, s_ed{};
      int bctype = 0;

      fin >> s_st[0] >> s_ed[0]
          >> s_st[1] >> s_ed[1]
          >> s_st[2] >> s_ed[2]
          >> bctype;
      require(fin.good(), "Bad read: region header block=" + std::to_string(nb0+1) +
                          " region=" + std::to_string(r+1));

      reg.bctype = bctype;
      reg.nbs = nb0 + 1; 
      reg.s_st = s_st;
      reg.s_ed = s_ed;

      // [核心修正] 自动计算几何属性
      if (s_st[0] == s_ed[0]) {
          reg.s_nd = 1; // i-face
          if (s_st[0] == 1)        reg.s_lr = -1; // Min Face
          else if (s_st[0] == imax) reg.s_lr =  1; // Max Face
          else reg.s_lr = 0; // Internal cut
      } else if (s_st[1] == s_ed[1]) {
          reg.s_nd = 2; // j-face
          if (s_st[1] == 1)        reg.s_lr = -1;
          else if (s_st[1] == jmax) reg.s_lr =  1;
          else reg.s_lr = 0;
      } else if (s_st[2] == s_ed[2]) {
          reg.s_nd = 3; // k-face
          if (s_st[2] == 1)        reg.s_lr = -1;
          else if (s_st[2] == kmax) reg.s_lr =  1;
          else reg.s_lr = 0;
      } else {
          reg.s_nd = 0; // Not a plane (Volume or Oblique)
          reg.s_lr = 0;
      }

      if (bctype < 0) {
        std::array<int,3> t_st{}, t_ed{};
        int nbt = 0, ibcwin = 0;

        fin >> t_st[0] >> t_ed[0]
            >> t_st[1] >> t_ed[1]
            >> t_st[2] >> t_ed[2]
            >> nbt >> ibcwin;
        require(fin.good(), "Bad read: connect info block=" + std::to_string(nb0+1) +
                            " region=" + std::to_string(r+1));

        reg.nbt = nbt;       
        reg.ibcwin = ibcwin;
        reg.t_st = t_st;
        reg.t_ed = t_ed;

        // [核心修正] 检查接口合法性
        if (reg.s_nd == 0) {
            throw std::runtime_error("Error: BC Region " + std::to_string(r+1) + 
                                     " in Block " + std::to_string(nb0+1) + 
                                     " is an Interface (type<0) but is NOT a flat plane!");
        }
        if (reg.ibcwin <= 0) {
             throw std::runtime_error("Error: BC Region " + std::to_string(r+1) + 
                                     " in Block " + std::to_string(nb0+1) + 
                                     " has invalid target window id (ibcwin <= 0)!");
        }
      }

      out.block_bc[nb0].regions[r] = reg;
    }
  }

  if (opt.verbose) {
    std::cout << "[BCReader] finished reading bc info\n";
  }
  return out;
}

} // namespace orion::bc