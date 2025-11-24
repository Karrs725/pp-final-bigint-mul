#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"

#define TILE_SIZE 32

// Optimized kernel using shared memory and tiling
__global__ void multiply_tiled_kernel(const i32* lhs, const i32* rhs, 
                                      i64* result, u32 len1, u32 len2) {
    __shared__ i32 s_lhs[TILE_SIZE];
    __shared__ i32 s_rhs[TILE_SIZE];
    
    u32 row = blockIdx.y * blockDim.y + threadIdx.y;
    u32 col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= len1 || col >= len2) return;
    
    i64 sum = 0;
    
    // Tiled multiplication with shared memory
    for (u32 tile = 0; tile < (len2 + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        if (threadIdx.y == 0 && col < len2) {
            s_rhs[threadIdx.x] = rhs[col];
        }
        if (threadIdx.x == 0 && row < len1) {
            s_lhs[threadIdx.y] = lhs[row];
        }
        __syncthreads();
        
        // Compute partial product
        if (row < len1 && col < len2) {
            sum = (i64)s_lhs[threadIdx.y] * (i64)s_rhs[threadIdx.x];
            
            // Accumulate to correct position using atomic operations
            u32 pos = row + col;
            atomicAdd(reinterpret_cast<unsigned long long*>(&result[pos]), 
                     static_cast<unsigned long long>(sum));
        }
        __syncthreads();
    }
}

// Single-pass carry propagation with 64-bit intermediate
__global__ void carry_propagate_64bit(i64* values, i32* result, u32 len) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < len) {
        i64 val = values[idx];
        result[idx] = val % 10;
        
        if (idx + 1 < len && val >= 10) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&values[idx + 1]), 
                     static_cast<unsigned long long>(val / 10));
        }
    }
}

class SchoolbookMul_GPU_Optimized : public BigMulImpl {
private:
    void checkCuda(cudaError_t result, const char* msg) const {
        if (result != cudaSuccess) {
            std::cerr << "CUDA Error: " << msg << " - " 
                      << cudaGetErrorString(result) << std::endl;
        }
    }

public:
    std::string name() const override { return "gpu-schoolbook-opt"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) const override {
        u32 len1 = lhs.size();
        u32 len2 = rhs.size();
        u32 result_len = len1 + len2;
        
        // Use 64-bit for intermediate results to avoid overflow
        i32 *d_lhs, *d_rhs, *d_result;
        i64 *d_result_64;
        
        checkCuda(cudaMalloc(&d_lhs, len1 * sizeof(i32)), "malloc lhs");
        checkCuda(cudaMalloc(&d_rhs, len2 * sizeof(i32)), "malloc rhs");
        checkCuda(cudaMalloc(&d_result, result_len * sizeof(i32)), "malloc result");
        checkCuda(cudaMalloc(&d_result_64, result_len * sizeof(i64)), "malloc result64");
        
        checkCuda(cudaMemcpy(d_lhs, lhs.data(), len1 * sizeof(i32), cudaMemcpyHostToDevice), "copy lhs");
        checkCuda(cudaMemcpy(d_rhs, rhs.data(), len2 * sizeof(i32), cudaMemcpyHostToDevice), "copy rhs");
        checkCuda(cudaMemset(d_result_64, 0, result_len * sizeof(i64)), "memset result64");
        
        // Launch tiled multiplication kernel
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((len2 + TILE_SIZE - 1) / TILE_SIZE, 
                    (len1 + TILE_SIZE - 1) / TILE_SIZE);
        
        multiply_tiled_kernel<<<blocks, threads>>>(d_lhs, d_rhs, d_result_64, len1, len2);
        checkCuda(cudaGetLastError(), "multiply kernel");
        checkCuda(cudaDeviceSynchronize(), "multiply sync");
        
        // Carry propagation (multiple passes if needed)
        u32 threads_per_block = 256;
        u32 num_blocks = (result_len + threads_per_block - 1) / threads_per_block;
        
        for (u32 pass = 0; pass < 5; pass++) {
            carry_propagate_64bit<<<num_blocks, threads_per_block>>>(d_result_64, d_result, result_len);
            checkCuda(cudaGetLastError(), "carry kernel");
            checkCuda(cudaDeviceSynchronize(), "carry sync");
        }
        
        // Copy result back
        BigInt result(result_len);
        checkCuda(cudaMemcpy(result.data(), d_result, result_len * sizeof(i32), cudaMemcpyDeviceToHost), "copy result");
        
        // Cleanup
        cudaFree(d_lhs);
        cudaFree(d_rhs);
        cudaFree(d_result);
        cudaFree(d_result_64);
        
        // Remove leading zeros
        while (result.size() > 1 && result.back() == 0) {
            result.pop_back();
        }
        
        return result;
    }
};

// Register implementation
static bool _ = [](){
    register_impl("gpu-schoolbook-opt", [](){ return new SchoolbookMul_GPU_Optimized(); });
    return true;
}();
