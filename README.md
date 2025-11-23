# Parallel Large Integer Multiplication

High-performance arbitrary-precision integer multiplication with CPU and GPU implementations.

## Quick Start

```bash
cmake -B build && cmake --build build -j
./build/bin/benchmark --list
./build/bin/benchmark --size 1000
```

## Features

- **CPU**: Schoolbook algorithm
- **GPU**: CUDA-accelerated implementations (basic + optimized)
- **Benchmarking**: Compare implementations with custom input sizes
- **Modular**: Registry-based plugin architecture

## Usage

```bash
./build/bin/benchmark [OPTIONS]

Options:
  --list              List available implementations
  --impl <name>       Run specific implementation
  --size <n>          Random number size (default: 1000)
  --file <path>       Read from file
  --help              Show help

Examples:
  ./build/bin/benchmark --size 5000
  ./build/bin/benchmark --impl gpu-schoolbook-opt --size 2000
```

## Requirements

- CMake 3.18+
- C++20 compiler
- CUDA Toolkit (optional, auto-detected)

## Project Structure

```
src/
  ├── cpu/           CPU implementations
  ├── gpu/           CUDA implementations
  └── registry.cpp   Plugin system
include/             Public API
benchmarks/          Benchmark driver
```

## Adding Implementations

1. Create implementation in `src/cpu/` or `src/gpu/`
2. Inherit from `BigMulImpl`
3. Register with `register_impl("name", creator)`
4. Rebuild - automatic discovery via CMake
