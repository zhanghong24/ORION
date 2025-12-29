# ORION

**ORION** is a high-performance, extensible CFD solver framework designed for
large-scale **multi-block structured grids** on modern **heterogeneous HPC systems**.

The project targets extreme computational efficiency and long-term extensibility,
supporting CPU, GPU, and future accelerator architectures (e.g. DCU),
while maintaining a clean separation between **physics, numerics, parallel runtime, and I/O**.

---

## Key Features

- **Multi-block structured grid framework**
  - Arbitrary block connectivity
  - Efficient halo exchange with MPI
  - Block-level scheduling and execution

- **Heterogeneous HPC ready**
  - Architecture-neutral design
  - Backend support for CPU / CUDA / HIP / DCU (planned)
  - GPU-first data layout and execution model

- **High-performance parallelism**
  - MPI + accelerator programming model
  - CUDA-aware / device-aware MPI
  - Overlap of computation and communication
  - Interior / boundary region decomposition

- **Extensible physics and numerics**
  - Compressible Navier–Stokes (NS)
  - RANS models (planned)
  - Modular flux, reconstruction, and time-integration schemes
  - Explicit and implicit time marching

- **Scalable I/O for post-processing**
  - Per-rank, per-block `.vts` output
  - Root-generated `.vtm` collection files
  - Solver–I/O decoupled design

---

## Design Philosophy

ORION is designed around several core principles:

1. **Block as the fundamental unit**
   - Computation, communication, and I/O are organized at block level.

2. **Separation of concerns**
   - Mesh, fields, physics, numerics, time integration, communication, and I/O
     are strictly decoupled.

3. **Performance transparency**
   - Data layout and execution order are explicit.
   - Memory movement is minimized and controllable.

4. **Long-term extensibility**
   - New equations, turbulence models, solvers, and hardware backends
     can be added without invasive changes.

---

## Software Architecture

High-level module organization:

ORION

├─ core # Types, memory, parallel runtime, utilities

├─ mesh # Block and multi-block structured grids

├─ field # Flow variables and data layout (SoA)

├─ physics # Governing equations and physical models

├─ numerics # Reconstruction, fluxes, operators

├─ time # Time integration (explicit / implicit)

├─ comm # Halo exchange and MPI communication

├─ solver # Block and multi-block solvers

└─ io # VTS / VTM output


Each module is designed to be independently extensible and testable.

---
## Parallel Execution Model

- **MPI rank** manages one or more structured blocks
- Each block:
  - Stores its own mesh, fields, and boundary conditions
  - Exchanges halo data with neighboring blocks
- Typical timestep flow:
  1. Compute interior region
  2. Asynchronous halo exchange
  3. Compute boundary region
  4. Time integration update

---

## I/O Strategy

- Each MPI rank writes: 
output/step_xxxxxx/rank_xxxx/block_yyyy.vts

- MPI root writes: output/step_xxxxxx/collection.vtm

- Post-processing is fully compatible with ParaView.

---

## Target Use Cases

- Large-scale aerodynamic simulations
- Turbulent flow research on structured grids
- HPC algorithm development for CFD
- Research and production runs on heterogeneous clusters

---

## Project Status

ORION is under active development.

Planned milestones:
- [ ] Core multi-block runtime
- [ ] NS equations on CPU and GPU
- [ ] Implicit time integration
- [ ] RANS turbulence models
- [ ] Multi-backend abstraction layer (CPU / CUDA / HIP / DCU)

---

## License

License to be determined.

---

## Contact

ORION is a research-oriented HPC CFD project.
For questions or collaboration, please contact the project maintainer.

