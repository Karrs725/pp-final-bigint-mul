#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include "modint.hpp"

class NTT_cpu_omp : public BigMulImpl {
public:
    std::string name() const override { return "ntt-cpu-omp"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override;

private:
    using field = MontgomeryModInt<998244353>;

    static constexpr field one_ = field(1);
    static constexpr field zero_ = field(0);

    std::vector<field> roots, iroots, carry, icarry, root, iroot;

    void set_root(u32 n);
    void dft(u32 n, field *a);
    void idft(u32 n, field *a);
};

BigInt NTT_cpu_omp::multiply(const BigInt &lhs, const BigInt &rhs) {
    const u32 lhs_len= lhs.size();
    const u32 rhs_len = rhs.size();

    if (lhs_len == 0 || rhs_len == 0) {
        return BigInt{};
    }

    const u32 result_len = std::max<u32>(4, lhs_len + rhs_len);

    u32 ntt_len = std::bit_ceil(result_len);

    std::vector<field> A(ntt_len, zero_);
    std::vector<field> B(ntt_len, zero_);

    #pragma omp parallel for
    for (u32 i = 0; i < ntt_len; i++) {
        if (i < lhs_len) A[i] = field(lhs[i]);
        if (i < rhs_len) B[i] = field(rhs[i]);
    }

    dft(ntt_len, A.data());
    dft(ntt_len, B.data());
    #pragma omp parallel for
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

void NTT_cpu_omp::set_root(u32 n) {
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

    root.reserve(n / 2);
    iroot.reserve(n / 2);
    root.push_back(one_);
    iroot.push_back(one_);
    for (u32 i = 0; i < n / 2 - 1; i++) {
        root.push_back(root[i] * carry[__builtin_ctz(~i)]);
        iroot.push_back(iroot[i] * icarry[__builtin_ctz(~i)]);
    }
}

void NTT_cpu_omp::dft(u32 n, field *a) {
    set_root(n);

    for (u32 len = n; len >= 2; len >>= 1) {
        u32 half = len >> 1;

        #pragma omp parallel for
        for (u32 k = 0; k < n / len; k++) {
            u32 i = k * len;
            field r = root[k];

            for (u32 j = 0; j < half; j++) {
                field x = a[i + j];
                field y = a[i + j + half] * r;
                a[i + j] = x + y;
                a[i + j + half] = x - y;
            }
        }
    }
}

void NTT_cpu_omp::idft(u32 n, field *a) {
    set_root(n);

    for (u32 len = 2; len <= n; len <<= 1) {
        u32 half = len >> 1;

        #pragma omp parallel for
        for (u32 k = 0; k < n / len; k++) {
            u32 i = k * len;
            field ir = iroot[k];

            for (u32 j = 0; j < half; j++) {
                field x = a[i + j];
                field y = a[i + j + half];
                a[i + j] = x + y;
                a[i + j + half] = (x - y) * ir;
            }
        }
    }

    #pragma omp parallel for
    for (u32 i = 0; i < n; i++) {
        a[i] /= n;
    }
}

static bool _ = [](){
    register_impl("ntt-cpu-omp", [](){ return new NTT_cpu_omp(); });
    return true;
}();