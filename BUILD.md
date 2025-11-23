# Building the Project

This project uses CMake. No build scripts needed!

## Quick Start

```bash
# Configure and build
cmake -B build
cmake --build build

# Run benchmark
./build/bin/benchmark --list
./build/bin/benchmark --size 1000
```

## Build Options

```bash
# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Clean rebuild
rm -rf build
cmake -B build
cmake --build build

# Parallel build (faster)
cmake --build build -j8
```

## Requirements

- CMake 3.18+
- C++23 compiler (GCC 12+, Clang 15+)
- Optional: CUDA toolkit (auto-detected)
- Optional: OpenCL (uncomment in CMakeLists.txt)

## Project Structure

```
src/           # Core implementation
├── cpu/       # CPU algorithms
└── gpu/       # GPU algorithms (CUDA, OpenCL)
include/       # Public headers
benchmarks/    # Benchmark code
build/         # Build output (generated)
└── bin/       # Executables
```

That's it!
