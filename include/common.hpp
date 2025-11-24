#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <fstream>
#include <thread>
#include <utility>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <random>
#include <assert.h>
#include <string_view>
#include <ranges>
#include <unordered_map>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using i32 = __int32_t;
using u32 = __uint32_t;
using i64 = __int64_t;
using u64 = __uint64_t;