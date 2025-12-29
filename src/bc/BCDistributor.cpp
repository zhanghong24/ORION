#include "bc/BCDistributor.hpp"
#include <mpi.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <limits>

namespace orion::bc {

static void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}

// -------------------- pack/unpack helpers --------------------
struct ByteWriter {
  std::vector<std::uint8_t> buf;

  template <typename T>
  void put_pod(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>, "put_pod requires trivially copyable type");
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
  }

  void put_bytes(const void* data, std::size_t nbytes) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(data);
    buf.insert(buf.end(), p, p + nbytes);
  }

  void put_string(const std::string& s) {
    require(s.size() <= (std::size_t)std::numeric_limits<std::int32_t>::max(),
            "BCDistributor: string too long");
    std::int32_t n = (std::int32_t)s.size();
    put_pod(n);
    if (n > 0) put_bytes(s.data(), (std::size_t)n);
  }
};

struct ByteReader {
  const std::uint8_t* p = nullptr;
  const std::uint8_t* e = nullptr;

  template <typename T>
  T get_pod() {
    static_assert(std::is_trivially_copyable_v<T>, "get_pod requires trivially copyable type");
    require(p + sizeof(T) <= e, "BCDistributor: ByteReader overflow");
    T v{};
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }

  void get_bytes(void* out, std::size_t nbytes) {
    require(p + nbytes <= e, "BCDistributor: ByteReader overflow bytes");
    std::memcpy(out, p, nbytes);
    p += nbytes;
  }

  std::string get_string() {
    std::int32_t n = get_pod<std::int32_t>();
    require(n >= 0, "BCDistributor: negative string length");
    require(p + (std::size_t)n <= e, "BCDistributor: string overflow");
    std::string s;
    s.resize((std::size_t)n);
    if (n > 0) get_bytes(s.data(), (std::size_t)n);
    return s;
  }
};

// [修正] pack_region 需要包含 s_nd 和 s_lr
static void pack_region(ByteWriter& w, const BCRegion& r)
{
  std::int32_t bctype = (std::int32_t)r.bctype;
  std::int32_t nbs    = (std::int32_t)r.nbs;
  std::int32_t nbt    = (std::int32_t)r.nbt;
  std::int32_t ibcwin = (std::int32_t)r.ibcwin;
  // [新增]
  std::int32_t s_nd   = (std::int32_t)r.s_nd;
  std::int32_t s_lr   = (std::int32_t)r.s_lr;

  w.put_pod(bctype);
  w.put_pod(nbs);

  // s_st/s_ed
  for (int k=0;k<3;++k) w.put_pod((std::int32_t)r.s_st[k]);
  for (int k=0;k<3;++k) w.put_pod((std::int32_t)r.s_ed[k]);

  // connect info
  w.put_pod(nbt);
  w.put_pod(ibcwin);
  for (int k=0;k<3;++k) w.put_pod((std::int32_t)r.t_st[k]);
  for (int k=0;k<3;++k) w.put_pod((std::int32_t)r.t_ed[k]);

  // [新增] 几何信息
  w.put_pod(s_nd);
  w.put_pod(s_lr);
}

// [修正] unpack_region 需要包含 s_nd 和 s_lr
static void unpack_region(ByteReader& rd, BCRegion& r)
{
  r.bctype = (int)rd.get_pod<std::int32_t>();
  r.nbs    = (int)rd.get_pod<std::int32_t>();

  for (int k=0;k<3;++k) r.s_st[k] = (int)rd.get_pod<std::int32_t>();
  for (int k=0;k<3;++k) r.s_ed[k] = (int)rd.get_pod<std::int32_t>();

  r.nbt    = (int)rd.get_pod<std::int32_t>();
  r.ibcwin = (int)rd.get_pod<std::int32_t>();

  for (int k=0;k<3;++k) r.t_st[k] = (int)rd.get_pod<std::int32_t>();
  for (int k=0;k<3;++k) r.t_ed[k] = (int)rd.get_pod<std::int32_t>();

  // [新增]
  r.s_nd = (int)rd.get_pod<std::int32_t>();
  r.s_lr = (int)rd.get_pod<std::int32_t>();
}

void distrib_bc_fast(BCData& bc,
                     int myrank, int nranks,
                     const BCDistribOptions& opt)
{
  (void)nranks;
  const int master = opt.master;

  std::vector<std::uint8_t> buffer;

  // ---------------- root packs ----------------
  if (myrank == master) {
    // sanity check
    require(bc.number_of_blocks >= 0, "BCDistributor: invalid number_of_blocks");
    require((int)bc.block_pid.size() == bc.number_of_blocks,
            "BCDistributor: block_pid size mismatch on root");
    require((int)bc.block_bc.size() == bc.number_of_blocks,
            "BCDistributor: block_bc size mismatch on root");

    ByteWriter w;

    // header
    w.put_pod((std::int32_t)bc.flow_solver_id);
    w.put_pod((std::int32_t)bc.nprocs_in_file);
    w.put_pod((std::int32_t)bc.number_of_blocks);

    // block_pid (1-based)
    for (int i=0;i<bc.number_of_blocks;++i) {
      w.put_pod((std::int32_t)bc.block_pid[i]);
    }

    // per block
    for (int nb=0; nb<bc.number_of_blocks; ++nb) {
      const auto& b = bc.block_bc[nb];
      w.put_string(b.blockname);

      const std::int32_t nreg = (std::int32_t)b.regions.size();
      w.put_pod(nreg);

      for (std::int32_t r=0; r<nreg; ++r) {
        pack_region(w, b.regions[(std::size_t)r]);
      }
    }

    buffer = std::move(w.buf);

    if (opt.verbose) {
      std::cout << "[BCDistributorFast] packed bytes=" << buffer.size() << "\n";
    }
  }

  // ---------------- broadcast size then bytes ----------------
  std::uint64_t nbytes = (myrank == master) ? (std::uint64_t)buffer.size() : 0ull;
  MPI_Bcast(&nbytes, 1, MPI_UINT64_T, master, MPI_COMM_WORLD);

  if (myrank != master) {
    buffer.resize((std::size_t)nbytes);
  }

  // Chunked Broadcast
  std::size_t remaining = (std::size_t)nbytes;
  std::size_t offset = 0;
  while (remaining > 0) {
    const int chunk = (remaining > (std::size_t)std::numeric_limits<int>::max())
                        ? std::numeric_limits<int>::max()
                        : (int)remaining;
    MPI_Bcast(buffer.data() + offset, chunk, MPI_BYTE, master, MPI_COMM_WORLD);
    offset += (std::size_t)chunk;
    remaining -= (std::size_t)chunk;
  }

  // ---------------- non-root unpacks ----------------
  if (myrank != master) {
    ByteReader rd{buffer.data(), buffer.data() + buffer.size()};

    bc.flow_solver_id     = (int)rd.get_pod<std::int32_t>();
    bc.nprocs_in_file     = (int)rd.get_pod<std::int32_t>();
    bc.number_of_blocks   = (int)rd.get_pod<std::int32_t>();

    require(bc.number_of_blocks >= 0, "BCDistributor: invalid number_of_blocks unpack");

    bc.block_pid.resize((std::size_t)bc.number_of_blocks);
    bc.block_bc.resize((std::size_t)bc.number_of_blocks);

    for (int i=0;i<bc.number_of_blocks;++i) {
      bc.block_pid[(std::size_t)i] = (int)rd.get_pod<std::int32_t>();
    }

    for (int nb=0; nb<bc.number_of_blocks; ++nb) {
      auto& b = bc.block_bc[(std::size_t)nb];
      b.blockname = rd.get_string();

      const std::int32_t nreg = rd.get_pod<std::int32_t>();
      require(nreg >= 0, "BCDistributor: negative nregions");
      b.nregions = (int)nreg;
      b.regions.resize((std::size_t)nreg);

      for (std::int32_t r=0; r<nreg; ++r) {
        unpack_region(rd, b.regions[(std::size_t)r]);
      }
    }

    require(rd.p == rd.e, "BCDistributor: trailing bytes (pack/unpack mismatch)");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (myrank == master) {
    std::cout << "[BCDistributorFast] Distribute the topology successfully! (single-bcast)\n";
  }
}

} // namespace orion::bc