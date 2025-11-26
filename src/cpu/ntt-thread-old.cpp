#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include "modint.hpp"

class NTT_cpu_thread_old : public BigMulImpl {
public:
    std::string name() const override { return "ntt-cpu-thread-old"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override;

    void config(const cli::CLI& cli) override;

private:
    using field = MontgomeryModInt<998244353>;

    static constexpr field one_ = field(1);
    static constexpr field zero_ = field(0);

    std::vector<field> roots, iroots, carry, icarry;
    std::vector<field> root_, iroot_;

    void set_root(u32 n);
    void dft(u32 n, field *a);
    void idft(u32 n, field *a);

    u32 num_threads_ = std::thread::hardware_concurrency() ?
                        static_cast<u32>(std::thread::hardware_concurrency()) : 4;

};

BigInt NTT_cpu_thread_old::multiply(const BigInt &lhs, const BigInt &rhs) {
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

void NTT_cpu_thread_old::set_root(u32 n) {
    assert((n & (n - 1)) == 0);

    if (root_.size() == n / 2) {
        return;
    }

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

    root_.resize(n / 2);
    iroot_.resize(n / 2);
    field cur = one_, icur = one_;
    for (u32 i = 0; i < n / 2; i++) {
        root_[i] = cur;
        iroot_[i] = icur;
        cur *= carry[__builtin_ctz(~i)];
        icur *= icarry[__builtin_ctz(~i)];
    }
}

void NTT_cpu_thread_old::dft(u32 n, field *a) {
    set_root(n);

    std::vector<std::thread> threads(num_threads_);
    // std::cout << "Using " << threads.size() << " threads for NTT.\n";

    for (u32 len = n; len >= 2; len >>= 1) {
        const u32 half = len >> 1;
        const field root = one_;

        auto work = [=](u32 l_m, u32 r_m) -> void {
            field w = root_[l_m];
            u32 l = l_m * len;
            for (u32 m = l_m; m < r_m; m++, l += len) {
                u32 r = l + len;
                for (u32 k = 0; k < half; k++) {
                    u32 i = l + k;
                    u32 j = l + k + half;
                    field x = a[i];
                    field y = a[j] * w;
                    a[i] = x + y;
                    a[j] = x - y;
                }
                w *= carry[__builtin_ctz(~m)];
            }
        };

        const u32 num_block = n / len;
        const u32 num_threads = std::min(num_block, num_threads_);
        const u32 block_per_thread = (num_block + num_threads - 1) / num_threads;

        for (u32 i = 0; i < num_threads; i++) {
            const u32 l_m = i * block_per_thread;
            const u32 r_m = std::min(num_block, (i + 1) * block_per_thread);
            threads[i] = std::thread(work, l_m, r_m);
        }

        for (u32 i = 0; i < num_threads; i++) {
            threads[i].join();
        }
    }
}

void NTT_cpu_thread_old::idft(u32 n, field *a) {
    set_root(n);

    std::vector<std::thread> threads(num_threads_);

    for (u32 len = 2; len <= n; len <<= 1) {
        const u32 half = len >> 1;

        auto work = [=](u32 l_m, u32 r_m) -> void {
            field w = iroot_[l_m];
            u32 l = l_m * len;
            for (u32 m = l_m; m < r_m; m++, l += len) {
                u32 r = l + len;
                for (u32 k = 0; k < half; k++) {
                    u32 i = l + k;
                    u32 j = l + k + half;
                    field x = a[i];
                    field y = a[j];
                    a[i] = x + y;
                    a[j] = (x - y) * w;
                }
                w *= icarry[__builtin_ctz(~m)];
            }
        };

        const u32 num_block = n / len;
        const u32 num_threads = std::min(num_block, num_threads_);
        const u32 block_per_thread = (num_block + num_threads - 1) / num_threads;

        for (u32 i = 0; i < num_threads; i++) {
            const u32 l_m = i * block_per_thread;
            const u32 r_m = std::min(num_block, (i + 1) * block_per_thread);
            threads[i] = std::thread(work, l_m, r_m);
        }

        for (u32 i = 0; i < num_threads; i++) {
            threads[i].join();
        }
    }

    for (u32 i = 0; i < n; i++) {
        a[i] /= n;
    }
}

void NTT_cpu_thread_old::config(const cli::CLI& cli) {
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
    register_impl("ntt-cpu-thread-old", [](){ return new NTT_cpu_thread_old(); });
    return true;
}();