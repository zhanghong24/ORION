# ============================================================
# Core library (architecture independent)
# ============================================================

add_library(orion_core
    core/Runtime.cpp
    core/Params.cpp
    core/OrionSolver.cpp
    mesh/Plot3DReader.cpp
    mesh/GridDistributor.cpp
    mesh/MetricComputer.cpp
    bc/BCReader.cpp
    bc/BCDistributor.cpp
    bc/BCPreprocess.cpp
    bc/PhysicalBC.cpp
    preprocess/FlowField.cpp
    preprocess/DebugDump.cpp
    preprocess/Inflow.cpp
    preprocess/InitialCondition.cpp
    solver/HaloExchanger.cpp
    solver/StateUpdater.cpp
    solver/StateUpdater.cpp
    solver/TimeIntegrator.cpp
    solver/FluxComputer.cpp
    solver/InviscidFluxComputer.cpp
    solver/ResidualMonitor.cpp
    postprocess/PostProcess.cpp
)

target_include_directories(orion_core PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(orion_core
    MPI::MPI_CXX
)

# ============================================================
# Backend libraries
# ============================================================

if(ORION_BACKEND STREQUAL "CPU")

    add_library(orion_backend_cpu
        backend/cpu/BackendCPU.cpp
    )

    target_compile_definitions(orion_backend_cpu PUBLIC ORION_BACKEND_CPU)

elseif(ORION_BACKEND STREQUAL "CUDA")

    add_library(orion_backend_cuda
        backend/cuda/BackendCUDA.cu
    )

    target_link_libraries(orion_backend_cuda
        CUDA::cudart
    )

endif()

# ============================================================
# Executable
# ============================================================

add_executable(orion_exec
    main.cpp
)

target_link_libraries(orion_exec
    orion_core
)

if(TARGET orion_backend_cpu)
    target_link_libraries(orion_exec orion_backend_cpu)
endif()

if(TARGET orion_backend_cuda)
    target_link_libraries(orion_exec orion_backend_cuda)
endif()
