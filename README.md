# parallel-large-integer-multiplication
```
big-mul-comparison/
├── CMakeLists.txt or build.py
├── docs/
│   ├── overview.md
│   └── algorithm-notes.md
├── scripts/
│   ├── gen_testcases.py
│   ├── run_all_benchmarks.py
│   └── plot_results.py
├── data/
│   ├── testcases/
│   └── benchmark_results/
├── include/
│   └── bigmul/
│       ├── bigint.hpp
│       ├── interface.hpp
│       ├── registry.hpp
│       └── util.hpp
├── src/
│   ├── cpu/
│   │   ├── schoolbook.cpp
│   │   ├── karatsuba.cpp
│   │   ├── toomcook.cpp
│   │   ├── fft.cpp
│   │   └── ntt.cpp
│   ├── gpu/
│   │   ├── cuda_fft.cu
│   │   ├── cuda_schoolbook.cu
│   │   ├── opencl_fft.cl
│   │   └── sycl_ntt.cpp
│   └── registry.cpp
├── tests/
│   ├── correctness_test.cpp
│   ├── random_test.cpp
│   └── edge_case_test.cpp
├── benchmark/
│   ├── bench_main.cpp
│   └── configs/
│       ├── small.json
│       ├── medium.json
│       └── large.json
└── results/
```