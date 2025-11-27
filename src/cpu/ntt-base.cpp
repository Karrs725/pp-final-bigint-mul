#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include "modint.hpp"
#include <omp.h>

class NTT_cpu_base : public BigMulImpl {
public:
    std::string name() const override { return "ntt-base"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override;

    void config(const cli::CLI& cli) override;

private:
    using field = MontgomeryModInt<998244353>;

    static constexpr field one_ = field(1);
    static constexpr field zero_ = field(0);

    std::vector<field> root_bit_, iroot_bit_, carry_, icarry_;
    std::vector<field> roots_, iroots_;

    void set_root(u32 n);
    
    template<bool inverse>
    void transform(u32 n, field *a);

    u32 num_threads_ = std::thread::hardware_concurrency() ?
                        static_cast<u32>(std::thread::hardware_concurrency()) : 4;
};

BigInt NTT_cpu_base::multiply(const BigInt &lhs, const BigInt &rhs) {
    const u32 lhs_len = lhs.size();
    const u32 rhs_len = rhs.size();

    if (lhs_len == 0 || rhs_len == 0) {
        return BigInt{};
    }

    omp_set_num_threads(num_threads_);

    const u32 result_len = std::max<u32>(4, lhs_len + rhs_len);
    u32 ntt_len = std::bit_ceil(result_len);

    set_root(ntt_len);

    std::vector<field> A(ntt_len, zero_);
    std::vector<field> B(ntt_len, zero_);

    // Parallel initialization
    // #pragma omp parallel for schedule(static)
    for (u32 i = 0; i < ntt_len; i++) {
        if (i < lhs_len) A[i] = field(lhs[i]);
        if (i < rhs_len) B[i] = field(rhs[i]);
    }

    transform<false>(ntt_len, A.data());
    transform<false>(ntt_len, B.data());

    // Parallel pointwise multiplication
    // #pragma omp parallel for schedule(static)
    for (u32 i = 0; i < ntt_len; i++) {
        A[i] *= B[i];
    }

    transform<true>(ntt_len, A.data());

    // Parallel division by n
    field inv_n = field(ntt_len).inv();
    // #pragma omp parallel for schedule(static)
    for (u32 i = 0; i < ntt_len; i++) {
        A[i] *= inv_n;
    }

    // Sequential carry propagation (cannot be parallelized easily)
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

void NTT_cpu_base::set_root(u32 n) {
    if (roots_.size() >= n / 2) {
        return;
    }

    const int h = std::__lg(n);
    root_bit_.resize(h - 1);
    iroot_bit_.resize(h - 1);

    root_bit_[h - 2] = field(field::get_primitive_root_prime()).pow((field::get_mod() - 1) / n);
    iroot_bit_[h - 2] = one_ / root_bit_[h - 2];
    for (int i = h - 3; i >= 0; i--) {
        root_bit_[i] = root_bit_[i + 1] * root_bit_[i + 1];
        iroot_bit_[i] = iroot_bit_[i + 1] * iroot_bit_[i + 1];
    }

    carry_.resize(h - 1);
    icarry_.resize(h - 1);
    field low = one_, ilow = one_;
    for (int i = 0; i < h - 1; i++) {
        carry_[i] = root_bit_[i] * ilow;
        icarry_[i] = iroot_bit_[i] * low;
        low *= root_bit_[i];
        ilow *= iroot_bit_[i];
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

template<bool inverse>
void NTT_cpu_base::transform(u32 n, field *a) {
    const std::vector<field> &roots = inverse ? iroots_ : roots_;

    if constexpr (inverse) {
        // IDFT: len goes 2 -> n
        for (u32 len = 2; len <= n; len <<= 1) {
            const u32 half = len >> 1;
            const u32 num_blocks = n / len;

            //#pragma omp parallel for schedule(static)
            for (u32 block = 0; block < num_blocks; block++) {
                field root = roots[block];
                field *x_ptr = a + block * len;
                field *y_ptr = x_ptr + half;

                for (u32 j = 0; j < half; j++) {
                    field x = x_ptr[j];
                    field y = y_ptr[j];
                    x_ptr[j] = x + y;
                    y_ptr[j] = (x - y) * root;
                }
            }
        }
    } else {
        // DFT: len goes n -> 2
        for (u32 len = n; len >= 2; len >>= 1) {
            const u32 half = len >> 1;
            const u32 num_blocks = n / len;

            // #pragma omp parallel for schedule(static)
            for (u32 block = 0; block < num_blocks; block++) {
                field root = roots[block];
                field *x_ptr = a + block * len;
                field *y_ptr = x_ptr + half;

                for (u32 j = 0; j < half; j++) {
                    field x = x_ptr[j];
                    field y = y_ptr[j] * root;
                    x_ptr[j] = x + y;
                    y_ptr[j] = x - y;
                }
            }
        }
    }
}

void NTT_cpu_base::config(const cli::CLI& cli) {
    if (cli.has_option("threads")) {
        try {
            auto parsed = std::stoul(cli.get_option("threads"));
            if (parsed > 0) num_threads_ = static_cast<u32>(parsed);
        } catch (...) {
            // ignore invalid value
        }
    }
}

static bool _ = [](){
    register_impl("ntt-base", [](){ return new NTT_cpu_base(); });
    return true;
}();