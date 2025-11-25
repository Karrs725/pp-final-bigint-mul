cmake_minimum_required(VERSION 3.18)
project(BigIntMultiply CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
add_compile_options(-O3 -march=native -Wall -Wextra -g)

# Find OpenMP and link it to targets instead of relying on compile-only flags.
find_package(OpenMP)

# Optional: CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Set appropriate GPU architectures
    file(GLOB CUDA_SOURCES src/gpu/*.cu)
    add_definitions(-DUSE_CUDA)    
    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message(STATUS "CUDA enabled with architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Collect all source files
file(GLOB CORE_SOURCES src/*.cpp)
file(GLOB CPU_SOURCES src/cpu/*.cpp)

# Benchmark executable - compile everything directly to ensure static initializers run
add_executable(benchmark 
    benchmarks/bench_main.cpp
    ${CORE_SOURCES}
    ${CPU_SOURCES}
    ${CUDA_SOURCES}
    # ${OPENCL_SOURCES}
)

target_include_directories(benchmark PRIVATE include)
set_target_properties(benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: linking OpenMP to benchmark")
    target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found: building without OpenMP support")
endif()
