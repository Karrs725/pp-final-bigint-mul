#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include "modint.hpp"

using field = MontgomeryModInt<998244353>;

// ============================================================================
// GPU Kernels
// ============================================================================

// DFT butterfly kernel: processes all blocks for a given stage in parallel
__global__ void ntt_dft_kernel(field *a, const field *roots, u32 n, u32 len) {
    u32 half = len >> 1;
    u32 num_blocks = n / len;
    
    // Each thread handles one butterfly operation
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 total_butterflies = num_blocks * half;
    
    if (tid >= total_butterflies) return;
    
    // Determine which block and which element within the block
    u32 block_id = tid / half;
    u32 j = tid % half;
    
    // Calculate indices
    u32 base = block_id * len;
    u32 i = base + j;
    u32 k = i + half;
    
    // Get the twiddle factor for this block
    field root = roots[block_id];
    
    field x = a[i];
    field y = a[k] * root;
    a[i] = x + y;
    a[k] = x - y;
}

// IDFT butterfly kernel
__global__ void ntt_idft_kernel(field *a, const field *iroots, u32 n, u32 len) {
    u32 half = len >> 1;
    u32 num_blocks = n / len;
    
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u32 total_butterflies = num_blocks * half;
    
    if (tid >= total_butterflies) return;
    
    u32 block_id = tid / half;
    u32 j = tid % half;
    
    u32 base = block_id * len;
    u32 i = base + j;
    u32 k = i + half;
    
    field iroot = iroots[block_id];
    
    field x = a[i];
    field y = a[k];
    a[i] = x + y;
    a[k] = (x - y) * iroot;
}

// Pointwise multiplication kernel
__global__ void pointwise_mul_kernel(field *a, const field *b, u32 n) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        a[tid] *= b[tid];
    }
}

// Scale by inverse of n
__global__ void scale_kernel(field *a, field inv_n, u32 n) {
    u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        a[tid] *= inv_n;
    }
}

// ============================================================================
// NTT_gpu class
// ============================================================================

class NTT_gpu : public BigMulImpl {
public:
    std::string name() const override { return "ntt-gpu"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override;

private:
    static constexpr field one_ = field(1);
    static constexpr field zero_ = field(0);
    static constexpr u32 BLOCK_SIZE = 256;

    std::vector<field> roots_, iroots_;
    std::vector<field> carry_, icarry_;

    void set_root(u32 n);
    void compute_stage_roots(u32 n, u32 len, std::vector<field> &stage_roots, bool inverse);
};

void NTT_gpu::set_root(u32 n) {
    if (roots_.size() >= n / 2) return;

    const int h = std::__lg(n);
    
    std::vector<field> root_bit(h - 1), iroot_bit(h - 1);
    root_bit[h - 2] = field(field::get_primitive_root_prime()).pow((field::get_mod() - 1) / n);
    iroot_bit[h - 2] = one_ / root_bit[h - 2];
    
    for (int i = h - 3; i >= 0; i--) {
        root_bit[i] = root_bit[i + 1] * root_bit[i + 1];
        iroot_bit[i] = iroot_bit[i + 1] * iroot_bit[i + 1];
    }

    carry_.resize(h - 1);
    icarry_.resize(h - 1);
    field low = one_, ilow = one_;
    for (int i = 0; i < h - 1; i++) {
        carry_[i] = root_bit[i] * ilow;
        icarry_[i] = iroot_bit[i] * low;
        low *= root_bit[i];
        ilow *= iroot_bit[i];
    }

    roots_.resize(n / 2);
    iroots_.resize(n / 2);
    field root = one_, iroot = one_;
    for (u32 i = 0; i < n / 2; i++) {
        roots_[i] = root;
        iroots_[i] = iroot;
        root *= carry_[__builtin_ctz(~i)];
        iroot *= icarry_[__builtin_ctz(~i)];
    }
}

void NTT_gpu::compute_stage_roots(u32 n, u32 len, std::vector<field> &stage_roots, bool inverse) {
    u32 num_blocks = n / len;
    stage_roots.resize(num_blocks);
    const std::vector<field> &roots = inverse ? iroots_ : roots_;
    for (u32 b = 0; b < num_blocks; b++) {
        stage_roots[b] = roots[b];
    }
}

BigInt NTT_gpu::multiply(const BigInt &lhs, const BigInt &rhs) {
    const u32 lhs_len = lhs.size();
    const u32 rhs_len = rhs.size();

    if (lhs_len == 0 || rhs_len == 0) {
        return BigInt{};
    }

    const u32 result_len = std::max<u32>(4, lhs_len + rhs_len);
    u32 ntt_len = std::bit_ceil(result_len);

    set_root(ntt_len);

    // Prepare host data
    std::vector<field> A(ntt_len, zero_);
    std::vector<field> B(ntt_len, zero_);

    for (u32 i = 0; i < lhs_len; i++) A[i] = field(lhs[i]);
    for (u32 i = 0; i < rhs_len; i++) B[i] = field(rhs[i]);

    // Allocate device memory
    field *d_a, *d_b, *d_roots;
    checkCuda(cudaMalloc(&d_a, ntt_len * sizeof(field)), "malloc d_a");
    checkCuda(cudaMalloc(&d_b, ntt_len * sizeof(field)), "malloc d_b");
    checkCuda(cudaMalloc(&d_roots, (ntt_len / 2) * sizeof(field)), "malloc d_roots");

    // Copy data to device
    checkCuda(cudaMemcpy(d_a, A.data(), ntt_len * sizeof(field), cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(d_b, B.data(), ntt_len * sizeof(field), cudaMemcpyHostToDevice), "copy B");

    // ========== DFT of A and B ==========
    std::vector<field> stage_roots;
    for (u32 len = ntt_len; len >= 2; len >>= 1) {
        u32 half = len >> 1;
        u32 num_blocks = ntt_len / len;
        u32 total_butterflies = num_blocks * half;
        
        compute_stage_roots(ntt_len, len, stage_roots, false);
        checkCuda(cudaMemcpy(d_roots, stage_roots.data(), num_blocks * sizeof(field), cudaMemcpyHostToDevice), "copy roots");
        
        u32 grid_size = (total_butterflies + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        ntt_dft_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_roots, ntt_len, len);
        ntt_dft_kernel<<<grid_size, BLOCK_SIZE>>>(d_b, d_roots, ntt_len, len);
    }
    cudaDeviceSynchronize();

    // ========== Pointwise multiplication ==========
    u32 grid_size = (ntt_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    pointwise_mul_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, ntt_len);

    // ========== IDFT of A ==========
    for (u32 len = 2; len <= ntt_len; len <<= 1) {
        u32 half = len >> 1;
        u32 num_blocks = ntt_len / len;
        u32 total_butterflies = num_blocks * half;
        
        compute_stage_roots(ntt_len, len, stage_roots, true);
        checkCuda(cudaMemcpy(d_roots, stage_roots.data(), num_blocks * sizeof(field), cudaMemcpyHostToDevice), "copy iroots");
        
        u32 grid = (total_butterflies + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ntt_idft_kernel<<<grid, BLOCK_SIZE>>>(d_a, d_roots, ntt_len, len);
    }

    // Scale by 1/n
    field inv_n = field(ntt_len).inv();
    scale_kernel<<<grid_size, BLOCK_SIZE>>>(d_a, inv_n, ntt_len);
    
    cudaDeviceSynchronize();

    // Copy result back
    checkCuda(cudaMemcpy(A.data(), d_a, ntt_len * sizeof(field), cudaMemcpyDeviceToHost), "copy result");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_roots);

    // Carry propagation (sequential on CPU)
    BigInt result(result_len);
    i64 carry = 0;
    for (u32 i = 0; i < result_len; i++) {
        carry += A[i].get();
        result[i] = carry % 10;
        carry /= 10;
    }
    assert(carry == 0);

    result.trim();
    return result;
}

static bool _ = [](){
    register_impl("ntt-gpu", [](){ return new NTT_gpu(); });
    return true;
}();