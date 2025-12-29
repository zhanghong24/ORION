#include "mesh/Plot3DReader.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <limits>
#include <sstream>
#include <cstring>

namespace orion::mesh {

static void require(bool cond, const std::string& msg) {
  if (!cond) throw std::runtime_error(msg);
}

class FortranSeqReader32 {
public:
  explicit FortranSeqReader32(const std::string& fn)
  : fin_(fn, std::ios::binary) {
    require(fin_.is_open(), "Failed to open grid file: " + fn);
  }

  // Read one record payload into a byte buffer
  std::vector<std::uint8_t> read_record_bytes() {
    std::uint32_t len1 = 0, len2 = 0;
    fin_.read(reinterpret_cast<char*>(&len1), sizeof(len1));
    require(fin_.good(), "EOF or bad read (record header)");

    std::vector<std::uint8_t> buf(len1);
    fin_.read(reinterpret_cast<char*>(buf.data()), len1);
    require(fin_.good(), "EOF or bad read (record payload)");

    fin_.read(reinterpret_cast<char*>(&len2), sizeof(len2));
    require(fin_.good(), "EOF or bad read (record footer)");

    require(len1 == len2, "Fortran record length mismatch");
    return buf;
  }

  template <typename T>
  T read_scalar() {
    auto b = read_record_bytes();
    require(b.size() == sizeof(T), "Record size mismatch for scalar");
    T v{};
    std::memcpy(&v, b.data(), sizeof(T));
    return v;
  }

  template <typename T>
  std::vector<T> read_array(std::size_t count) {
    auto b = read_record_bytes();
    require(b.size() == count * sizeof(T), "Record size mismatch for array");
    std::vector<T> a(count);
    std::memcpy(a.data(), b.data(), b.size());
    return a;
  }

  // Read record of N elements but element type unknown (float/double).
  // Return as double.
  std::vector<double> read_real_as_double(std::size_t count) {
    auto b = read_record_bytes();
    const std::size_t bytes = b.size();
    const std::size_t expect_f = count * sizeof(float);
    const std::size_t expect_d = count * sizeof(double);

    if (bytes == expect_d) {
      std::vector<double> a(count);
      std::memcpy(a.data(), b.data(), bytes);
      return a;
    }
    if (bytes == expect_f) {
      std::vector<float> tmp(count);
      std::memcpy(tmp.data(), b.data(), bytes);
      std::vector<double> a(count);
      for (std::size_t i=0;i<count;++i) a[i] = static_cast<double>(tmp[i]);
      return a;
    }

    throw std::runtime_error("Unsupported real record size: bytes=" + std::to_string(bytes) +
                             " (count=" + std::to_string(count) + ")");
  }

private:
  std::ifstream fin_;
};

static MultiBlockGrid read_binary_fortran(const std::string& filename, int ndim, bool verbose) {
  require(ndim == 3, "Currently only ndim=3 supported (matches your Fortran path)");

  FortranSeqReader32 r(filename);

  MultiBlockGrid g;
  g.ndim = ndim;

  // record1: nblocks (int)
  const int nblocks = r.read_scalar<int>();
  require(nblocks > 0, "Invalid nblocks");
  g.nblocks = nblocks;
  g.blocks.resize(nblocks);

  if (verbose) {
    std::cout << "[Plot3DReader] read mesh filename: " << filename << "\n";
    std::cout << "[Plot3DReader] num of blocks: " << nblocks << "\n";
  }

  // next nblocks records: (idim,jdim,kdim)
  for (int nb=0; nb<nblocks; ++nb) {
    auto dims = r.read_array<int>(3);
    g.blocks[nb].idim = dims[0];
    g.blocks[nb].jdim = dims[1];
    g.blocks[nb].kdim = dims[2];

    require(g.blocks[nb].idim > 0 && g.blocks[nb].jdim > 0 && g.blocks[nb].kdim > 0,
            "Invalid block dims");

    g.nmax = std::max(g.nmax, std::max(g.blocks[nb].idim, std::max(g.blocks[nb].jdim, g.blocks[nb].kdim)));
    g.total_points += g.blocks[nb].npoints();

    if (verbose) {
      std::cout << "  block " << (nb+1) << ": "
                << g.blocks[nb].idim << " "
                << g.blocks[nb].jdim << " "
                << g.blocks[nb].kdim << "\n";
    }
  }

  // for each block: one record containing x then y then z (most likely)
  for (int nb=0; nb<nblocks; ++nb) {
    const int id = g.blocks[nb].idim;
    const int jd = g.blocks[nb].jdim;
    const int kd = g.blocks[nb].kdim;
    const std::size_t n = static_cast<std::size_t>(id)*jd*kd;

    // read one record, but it contains 3*n reals
    // We read as bytes and decide float/double by record size.
    // Reuse read_real_as_double with count=3*n.
    std::vector<double> xyz = r.read_real_as_double(3*n);

    // allocate (i-fast): (id,jd,kd)
    g.blocks[nb].x.resize(id, jd, kd);
    g.blocks[nb].y.resize(id, jd, kd);
    g.blocks[nb].z.resize(id, jd, kd);

    // split
    // Fortran write order matched our linear layout: i fastest, then j, then k.
    std::memcpy(g.blocks[nb].x.hostPtr(), xyz.data() + 0*n, n*sizeof(double));
    std::memcpy(g.blocks[nb].y.hostPtr(), xyz.data() + 1*n, n*sizeof(double));
    std::memcpy(g.blocks[nb].z.hostPtr(), xyz.data() + 2*n, n*sizeof(double));
  }

  if (verbose) {
    std::cout << "[Plot3DReader] Total num of grids: " << g.total_points << "\n";
  }
  return g;
}

// Very simple ASCII fallback that matches the same logical content:
// $nblocks
// dims list
// then x y z as plain numbers (3*n per block)
static MultiBlockGrid read_ascii_simple(const std::string& filename, int ndim, bool verbose) {
  require(ndim == 3, "ASCII fallback currently only ndim=3");
  std::ifstream fin(filename);
  require(fin.is_open(), "Failed to open ASCII grid file: " + filename);

  MultiBlockGrid g;
  g.ndim = ndim;

  int nblocks = 0;
  fin >> nblocks;
  require(nblocks > 0, "Invalid nblocks in ASCII");
  g.nblocks = nblocks;
  g.blocks.resize(nblocks);

  if (verbose) {
    std::cout << "[Plot3DReader] (ASCII) num of blocks: " << nblocks << "\n";
  }

  for (int nb=0; nb<nblocks; ++nb) {
    fin >> g.blocks[nb].idim >> g.blocks[nb].jdim >> g.blocks[nb].kdim;
    require(fin.good(), "Bad read dims in ASCII");
    g.nmax = std::max(g.nmax, std::max(g.blocks[nb].idim, std::max(g.blocks[nb].jdim, g.blocks[nb].kdim)));
    g.total_points += g.blocks[nb].npoints();
  }

  for (int nb=0; nb<nblocks; ++nb) {
    const int id = g.blocks[nb].idim;
    const int jd = g.blocks[nb].jdim;
    const int kd = g.blocks[nb].kdim;
    const std::size_t n = static_cast<std::size_t>(id)*jd*kd;

    g.blocks[nb].x.resize(id, jd, kd);
    g.blocks[nb].y.resize(id, jd, kd);
    g.blocks[nb].z.resize(id, jd, kd);

    for (std::size_t i=0;i<n;++i) fin >> g.blocks[nb].x.hostPtr()[i];
    for (std::size_t i=0;i<n;++i) fin >> g.blocks[nb].y.hostPtr()[i];
    for (std::size_t i=0;i<n;++i) fin >> g.blocks[nb].z.hostPtr()[i];

    require(fin.good(), "Bad read xyz in ASCII");
  }

  if (verbose) {
    std::cout << "[Plot3DReader] (ASCII) Total num of grids: " << g.total_points << "\n";
  }
  return g;
}

MultiBlockGrid Plot3DReader::read(const std::string& filename, int ndim,
                                  const Plot3DReaderOptions& opt)
{
  // 先尝试 binary Fortran sequential
  try {
    return read_binary_fortran(filename, ndim, opt.verbose);
  } catch (const std::exception& e) {
    if (!opt.try_ascii_fallback) throw;
    if (opt.verbose) {
      std::cerr << "[Plot3DReader] binary read failed, try ASCII fallback. Reason: "
                << e.what() << "\n";
    }
    return read_ascii_simple(filename, ndim, opt.verbose);
  }
}

} // namespace orion::mesh
