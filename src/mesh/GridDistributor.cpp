#include "mesh/GridDistributor.hpp"
#include <mpi.h>
#include <vector>
#include <list>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace orion::mesh {

static void require(bool ok, const std::string& msg) {
  if (!ok) throw std::runtime_error(msg);
}

// -------------------- pack/unpack helpers --------------------
struct ByteWriter {
  std::vector<std::uint8_t> buf;

  template <typename T>
  void put_pod(const T& v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    buf.insert(buf.end(), p, p + sizeof(T));
  }

  void put_bytes(const void* data, std::size_t nbytes) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(data);
    buf.insert(buf.end(), p, p + nbytes);
  }

  std::size_t size() const { return buf.size(); }
};

struct ByteReader {
  const std::uint8_t* p = nullptr;
  const std::uint8_t* e = nullptr;

  template <typename T>
  T get_pod() {
    require(p + sizeof(T) <= e, "ByteReader overflow");
    T v{};
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }

  void get_bytes(void* out, std::size_t nbytes) {
    require(p + nbytes <= e, "ByteReader overflow bytes");
    std::memcpy(out, p, nbytes);
    p += nbytes;
  }
};

// -------------------- main function --------------------
void distrib_grid_fast(MultiBlockGrid& grid,
                       std::vector<int>& block_pid_1based,
                       int myrank, int nranks,
                       const GridDistribOptions& opt)
{
  const int master = opt.master;
  const int TAG_BASE = 600100;  // base tag for chunks

  // -------- 1) Bcast meta (nblocks, nmax, dims, pids) --------
  int nblocks = (myrank == master) ? grid.nblocks : 0;
  int nmax    = (myrank == master) ? grid.nmax    : 0;

  MPI_Bcast(&nblocks, 1, MPI_INT, master, MPI_COMM_WORLD);
  MPI_Bcast(&nmax,    1, MPI_INT, master, MPI_COMM_WORLD);

  std::vector<int> dims; // nblocks*3
  if (myrank == master) {
    require((int)grid.blocks.size() == nblocks, "grid.blocks size mismatch on master");
    require((int)block_pid_1based.size() == nblocks, "block_pid size mismatch on master");

    dims.resize(nblocks * 3);
    for (int nb=0; nb<nblocks; ++nb) {
      dims[3*nb+0] = grid.blocks[nb].idim;
      dims[3*nb+1] = grid.blocks[nb].jdim;
      dims[3*nb+2] = grid.blocks[nb].kdim;
    }
  } else {
    grid.clear();
    grid.ndim = 3;
    grid.nblocks = nblocks;
    grid.nmax = nmax;
    grid.blocks.resize(nblocks);

    block_pid_1based.resize(nblocks);
    dims.resize(nblocks * 3);
  }

  MPI_Bcast(dims.data(), nblocks*3, MPI_INT, master, MPI_COMM_WORLD);
  MPI_Bcast(block_pid_1based.data(), nblocks, MPI_INT, master, MPI_COMM_WORLD);

  grid.nblocks = nblocks;
  grid.nmax = nmax;
  if ((int)grid.blocks.size() != nblocks) grid.blocks.resize(nblocks);
  for (int nb=0; nb<nblocks; ++nb) {
    grid.blocks[nb].idim = dims[3*nb+0];
    grid.blocks[nb].jdim = dims[3*nb+1];
    grid.blocks[nb].kdim = dims[3*nb+2];
  }

  // -------- 2) build ownership lists on master --------
  std::vector<std::vector<int>> owned_by_rank;
  if (myrank == master) {
    owned_by_rank.resize(nranks);
    for (int nb=0; nb<nblocks; ++nb) {
      const int owner = block_pid_1based[nb] - 1; // 1-based -> rank
      require(owner >= 0 && owner < nranks, "owner rank out of range");
      owned_by_rank[owner].push_back(nb);
    }
  }

  // -------- 3) MASTER: pack into chunks and Isend --------
  std::vector<MPI_Request> reqs;
  // [CRITICAL FIX]: Buffer storage to keep data alive during async send
  // We use std::list because pointers to elements remain valid after insertion/deletion
  std::list<std::vector<std::uint8_t>> keep_alive_buffers;

  std::vector<std::vector<int>> sent_blocks_by_rank; 
  std::vector<int> chunks_per_rank;

  if (myrank == master) {
    sent_blocks_by_rank.resize(nranks);
    chunks_per_rank.assign(nranks, 0);

    for (int r=0; r<nranks; ++r) {
      if (r == master) continue;

      const auto& list = owned_by_rank[r];
      std::size_t idx = 0;
      int chunk_id = 0;

      while (idx < list.size()) {
        ByteWriter w;
        
        // Header placeholder
        w.buf.reserve(std::min<std::size_t>(opt.max_chunk_bytes, 1ull<<20));
        w.put_pod<int>(0); 
        int count = 0;

        // Fill chunk
        while (idx < list.size()) {
          int nb = list[idx];
          const int id = grid.blocks[nb].idim;
          const int jd = grid.blocks[nb].jdim;
          const int kd = grid.blocks[nb].kdim;
          const std::size_t n = (std::size_t)id * jd * kd;

          const std::size_t need = sizeof(int) + 3 * n * sizeof(double);

          // Check size limit
          if (w.size() + need > opt.max_chunk_bytes && count > 0) break;

          // Pack block
          w.put_pod<int>(nb + 1);
          w.put_bytes(grid.blocks[nb].x.hostPtr(), n * sizeof(double));
          w.put_bytes(grid.blocks[nb].y.hostPtr(), n * sizeof(double));
          w.put_bytes(grid.blocks[nb].z.hostPtr(), n * sizeof(double));

          sent_blocks_by_rank[r].push_back(nb);
          ++count;
          ++idx;
        }

        // Patch count
        std::memcpy(w.buf.data(), &count, sizeof(int));

        // [CRITICAL FIX]: Move buffer to persistent storage before Isend
        keep_alive_buffers.push_back(std::move(w.buf));
        std::vector<std::uint8_t>& valid_buf = keep_alive_buffers.back();

        // Send
        MPI_Request rq{};
        const int tag = TAG_BASE + chunk_id; 
        MPI_Isend(valid_buf.data(), (int)valid_buf.size(), MPI_BYTE, r, tag, MPI_COMM_WORLD, &rq);
        reqs.push_back(rq);

        ++chunk_id;
      }
      chunks_per_rank[r] = chunk_id;
    }
  }

  // broadcast chunks count
  if (myrank != master) chunks_per_rank.assign(nranks, 0);
  MPI_Bcast(chunks_per_rank.data(), nranks, MPI_INT, master, MPI_COMM_WORLD);

  // -------- 4) NON-ROOT: receive chunks and unpack --------
  if (myrank != master) {
    const int nchunks = chunks_per_rank[myrank];
    for (int chunk_id=0; chunk_id<nchunks; ++chunk_id) {
      const int tag = TAG_BASE + chunk_id;

      MPI_Status st{};
      MPI_Probe(master, tag, MPI_COMM_WORLD, &st);

      int nbytes = 0;
      MPI_Get_count(&st, MPI_BYTE, &nbytes);
      require(nbytes > 0, "Received empty chunk");

      std::vector<std::uint8_t> buf((std::size_t)nbytes);
      MPI_Recv(buf.data(), nbytes, MPI_BYTE, master, tag, MPI_COMM_WORLD, &st);

      ByteReader rd{buf.data(), buf.data() + buf.size()};
      const int nblk_in_chunk = rd.get_pod<int>();

      for (int t=0; t<nblk_in_chunk; ++t) {
        const int nb1 = rd.get_pod<int>();
        const int nb = nb1 - 1;

        const int id = grid.blocks[nb].idim;
        const int jd = grid.blocks[nb].jdim;
        const int kd = grid.blocks[nb].kdim;
        const std::size_t n = (std::size_t)id * jd * kd;

        grid.blocks[nb].x.resize(id, jd, kd);
        rd.get_bytes(grid.blocks[nb].x.hostPtr(), n * sizeof(double));

        grid.blocks[nb].y.resize(id, jd, kd);
        rd.get_bytes(grid.blocks[nb].y.hostPtr(), n * sizeof(double));

        grid.blocks[nb].z.resize(id, jd, kd);
        rd.get_bytes(grid.blocks[nb].z.hostPtr(), n * sizeof(double));
      }
    }
  }

  // -------- 5) MASTER cleanup --------
  if (myrank == master) {
    if (!reqs.empty()) {
      MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }
    // Now safe to clear buffers (std::list auto clears)
    keep_alive_buffers.clear();

    // Release sent blocks memory
    for (int r=0; r<nranks; ++r) {
      if (r == master) continue;
      for (int nb : sent_blocks_by_rank[r]) {
        grid.blocks[nb].x.clear();
        grid.blocks[nb].y.clear();
        grid.blocks[nb].z.clear();
      }
    }
    if (opt.verbose) std::cout << "[GridDistributor] Done.\n";
  }
}

} // namespace orion::mesh
