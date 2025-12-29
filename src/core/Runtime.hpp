// src/core/Runtime.h
#pragma once

namespace orion::core {

struct Runtime {
  // MPI basic info
  static int myid;
  static int nproc;

  // initialization / finalization
  static void init(int argc, char** argv);
  static void barrier();
  static void finalize();
  static int rank() {return myid;}
  static int size() {return nproc;}

  // helpers
  static bool is_root() { return myid == 0; }
};

} // namespace orion
