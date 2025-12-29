cmake_minimum_required(VERSION 3.20)

project(ORION
    VERSION 0.1.0
    LANGUAGES C CXX
)

# ============================================================
# Global options
# ============================================================

option(USE_CUDA "Enable CUDA backend" OFF)
option(USE_HIP  "Enable HIP backend"  OFF)
option(USE_DCU  "Enable DCU backend"  OFF)

# Safety check: only one backend allowed
set(_cuda 0)
set(_hip  0)
set(_dcu  0)

if(USE_CUDA)
  set(_cuda 1)
endif()
if(USE_HIP)
  set(_hip 1)
endif()
if(USE_DCU)
  set(_dcu 1)
endif()

math(EXPR BACKEND_COUNT "${_cuda}+${_hip}+${_dcu}")

if(BACKEND_COUNT GREATER 1)
  message(FATAL_ERROR "Only one backend can be enabled at a time.")
endif()


# ============================================================
# C++ standard & basic flags
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================
# MPI (required)
# ============================================================

find_package(MPI REQUIRED)
message(STATUS "MPI found: ${MPI_CXX_COMPILER}")

# ============================================================
# Backend selection
# ============================================================

set(ORION_BACKEND "CPU")

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(ORION_BACKEND "CUDA")
    add_definitions(-DORION_USE_CUDA)
elseif(USE_HIP)
    # ---- Placeholder ----
    set(ORION_BACKEND "HIP")
    add_definitions(-DORION_USE_HIP)
elseif(USE_DCU)
    # ---- Placeholder ----
    set(ORION_BACKEND "DCU")
    add_definitions(-DORION_USE_DCU)
else()
    add_definitions(-DORION_USE_CPU)
endif()

message(STATUS "ORION backend: ${ORION_BACKEND}")

# ============================================================
# Include directories
# ============================================================

include_directories(
    ${PROJECT_SOURCE_DIR}/src
)

# ============================================================
# Subdirectories
# ============================================================

add_subdirectory(src)

