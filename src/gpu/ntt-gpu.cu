#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include "modint.hpp"

class NTT_gpu : public BigMulImpl {
public:
    std::string name() const override { return "ntt-gpu"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override;

private:
    using field = MontgomeryModInt<998244353>;

    static constexpr field one_ = field(1);
    static constexpr field zero_ = field(0);

    std::vector<field> roots, iroots, carry, icarry;

    void set_root(u32 n);
    void dft(u32 n, field *a);
    void idft(u32 n, field *a);
};

BigInt NTT_gpu::multiply(const BigInt &lhs, const BigInt &rhs) {
    const u32 lhs_len= lhs.size();
    const u32 rhs_len = rhs.size();

    if (lhs_len == 0 || rhs_len == 0) {
        return BigInt{};
    }

    const u32 result_len = std::max<u32>(4, lhs_len + rhs_len);

    u32 ntt_len = std::bit_ceil(result_len);

    std::vector<field> A(ntt_len, zero_);
    std::vector<field> B(ntt_len, zero_);

    for (u32 i = 0; i < ntt_len; i++) {
        if (i < lhs_len) A[i] = field(lhs[i]);
        if (i < rhs_len) B[i] = field(rhs[i]);
    }

    dft(ntt_len, A.data());
    dft(ntt_len, B.data());
    for (u32 i = 0; i < ntt_len; i++) {
        A[i] *= B[i];
    }
    idft(ntt_len, A.data());

    BigInt result(result_len);
    i32 carry = 0;
    for (u32 i = 0; i < result_len; i++) {
        carry += A[i].get();
        result[i] = carry % 10;
        carry /= 10;
    }
    assert(carry == 0);

    result.trim();

    return result;
}

void NTT_gpu::set_root(u32 n) {
    assert((n & (n - 1)) == 0);

    const int h = std::__lg(n);
    roots.resize(h - 1);
    iroots.resize(h - 1);

    roots[h - 2] = field(field::get_primitive_root_prime()).pow((field::get_mod() - 1) / n);
    iroots[h - 2] = one_ / roots[h - 2];

    for (int i = h - 3; i >= 0; i--) {
        roots[i] = roots[i + 1] * roots[i + 1];
        iroots[i] = iroots[i + 1] * iroots[i + 1];
    }

    carry.resize(h - 1);
    icarry.resize(h - 1);
    field low = one_, ilow = one_;
    for (int i = 0; i < h - 1; i++) {
        carry[i] = roots[i] * ilow;
        icarry[i] = iroots[i] * low;
        low *= roots[i];
        ilow *= iroots[i];
    }
}

namespace ntt_gpu_internal {

using field = MontgomeryModInt<998244353>;
__global__ void ntt_kernel(const field &root, field *p) {
    u32 i = threadIdx.x;
    u32 j = i + blockDim.x;
    field x = p[i];
    field y = p[j] * root;
    p[i] = x + y;
    p[j] = x - y;
}

} // namespace ntt_gpu_internal

void NTT_gpu::dft(u32 n, field *a) {
    set_root(n);

    field *d_a;
    checkCuda(cudaMalloc(&d_a, n * sizeof(field)), "malloc d_a");
    checkCuda(cudaMemcpy(d_a, a, n * sizeof(field), cudaMemcpyHostToDevice), "copy to d_a");

    for (u32 len = n; len >= 2; len >>= 1) {
        u32 half = len >> 1;
        field root = one_;
        for (u32 i = 0, m = 0; i < n; i += len, m++) {

            field *p = d_a + i;
            ntt_gpu_internal::ntt_kernel<<<1, half>>>(root, p);

            // for (u32 j = 0; j < half; j++) {
            //     field x = p[j];
            //     field y = p[j + half] * root;
            //     a[i + j] = x + y;
            //     a[i + j + half] = x - y;
            // }
            root *= carry[__builtin_ctz(~m)];
        }

        cudaDeviceSynchronize();
    }

    checkCuda(cudaMemcpy(a, d_a, n * sizeof(field), cudaMemcpyDeviceToHost), "copy from d_a");
    cudaFree(d_a);

}

void NTT_gpu::idft(u32 n, field *a) {
    set_root(n);

    for (u32 len = 2; len <= n; len <<= 1) {
        u32 half = len >> 1;
        field iroot = one_;
        for (u32 i = 0, m = 0; i < n; i += len, m++) {
            for (u32 j = 0; j < half; j++) {
                field x = a[i + j];
                field y = a[i + j + half];
                a[i + j] = x + y;
                a[i + j + half] = (x - y) * iroot;
            }
            iroot *= icarry[__builtin_ctz(~m)];
        }
    }

    for (u32 i = 0; i < n; i++) {
        a[i] /= n;
    }
}

static bool _ = [](){
    register_impl("ntt-gpu", [](){ return new NTT_gpu(); });
    return true;
}();