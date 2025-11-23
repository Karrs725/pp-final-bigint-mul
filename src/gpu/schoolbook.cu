#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for parallel multiplication (Phase 1: compute products)
__global__ void multiply_kernel(const i32* lhs, const i32* rhs, i32* result, 
                                u32 len1, u32 len2, u32 result_len) {
    // Each thread computes one element: result[k] += sum of lhs[i]*rhs[j] where i+j=k
    u32 k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (k < result_len) {
        i32 sum = 0;
        
        // For position k, we need all pairs (i,j) where i+j=k
        u32 i_start = (k >= len2) ? k - len2 + 1 : 0;
        u32 i_end = (k < len1) ? k : len1 - 1;
        
        for (u32 i = i_start; i <= i_end; i++) {
            u32 j = k - i;
            sum += lhs[i] * rhs[j];
        }
        
        result[k] = sum;
    }
}

// CUDA kernel for carry propagation (Phase 2)
__global__ void carry_propagate_kernel(i32* result, u32 len, i32* carry_out) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < len) {
        i32 val = result[idx];
        result[idx] = val % 10;
        
        if (idx + 1 < len) {
            atomicAdd(&result[idx + 1], val / 10);
        } else if (val >= 10) {
            atomicAdd(carry_out, val / 10);
        }
    }
}

// Optimized carry propagation using parallel scan
__global__ void carry_scan_kernel(i32* result, i32* carries, u32 len) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < len) {
        i32 val = result[idx];
        carries[idx] = val / 10;
        result[idx] = val % 10;
    }
}

__global__ void apply_carries_kernel(i32* result, i32* carries, u32 len) {
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx > 0 && idx < len) {
        result[idx] += carries[idx - 1];
    }
}

class SchoolbookMul_GPU : public BigMulImpl {
private:
    // Check for CUDA errors
    void checkCuda(cudaError_t result, const char* msg) {
        if (result != cudaSuccess) {
            std::cerr << "CUDA Error: " << msg << " - " 
                      << cudaGetErrorString(result) << std::endl;
        }
    }

public:
    std::string name() const override { return "gpu-schoolbook"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override {
        u32 len1 = lhs.size();
        u32 len2 = rhs.size();
        u32 result_len = len1 + len2;
        
        // Allocate device memory
        i32 *d_lhs, *d_rhs, *d_result;
        checkCuda(cudaMalloc(&d_lhs, len1 * sizeof(i32)), "malloc lhs");
        checkCuda(cudaMalloc(&d_rhs, len2 * sizeof(i32)), "malloc rhs");
        checkCuda(cudaMalloc(&d_result, result_len * sizeof(i32)), "malloc result");
        
        // Copy input to device
        checkCuda(cudaMemcpy(d_lhs, lhs.data(), len1 * sizeof(i32), cudaMemcpyHostToDevice), "copy lhs");
        checkCuda(cudaMemcpy(d_rhs, rhs.data(), len2 * sizeof(i32), cudaMemcpyHostToDevice), "copy rhs");
        checkCuda(cudaMemset(d_result, 0, result_len * sizeof(i32)), "memset result");
        
        // Launch multiplication kernel
        u32 threads_per_block = 256;
        u32 blocks = (result_len + threads_per_block - 1) / threads_per_block;
        
        multiply_kernel<<<blocks, threads_per_block>>>(d_lhs, d_rhs, d_result, 
                                                        len1, len2, result_len);
        checkCuda(cudaGetLastError(), "multiply kernel launch");
        checkCuda(cudaDeviceSynchronize(), "multiply kernel sync");
        
        // Iterative carry propagation (simple but works)
        // Note: This is still sequential but happens on GPU with fewer iterations
        for (u32 iter = 0; iter < 10; iter++) {  // Usually converges in 2-3 iterations
            i32 *d_carry_out;
            checkCuda(cudaMalloc(&d_carry_out, sizeof(i32)), "malloc carry");
            checkCuda(cudaMemset(d_carry_out, 0, sizeof(i32)), "memset carry");
            
            carry_propagate_kernel<<<blocks, threads_per_block>>>(d_result, result_len, d_carry_out);
            checkCuda(cudaGetLastError(), "carry kernel launch");
            checkCuda(cudaDeviceSynchronize(), "carry kernel sync");
            
            i32 carry;
            checkCuda(cudaMemcpy(&carry, d_carry_out, sizeof(i32), cudaMemcpyDeviceToHost), "copy carry");
            cudaFree(d_carry_out);
            
            if (carry == 0) break;
        }
        
        // Copy result back to host
        BigInt result(result_len);
        checkCuda(cudaMemcpy(result.data(), d_result, result_len * sizeof(i32), cudaMemcpyDeviceToHost), "copy result");
        
        // Cleanup
        cudaFree(d_lhs);
        cudaFree(d_rhs);
        cudaFree(d_result);
        
        // Remove leading zeros
        while (result.size() > 1 && result.back() == 0) {
            result.pop_back();
        }
        
        return result;
    }
};

// Register implementation
static bool _ = [](){
    register_impl("gpu-schoolbook", [](){ return new SchoolbookMul_GPU(); });
    return true;
}();
