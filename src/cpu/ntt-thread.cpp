#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include "modint.hpp"

class NTT_cpu_thread : public BigMulImpl {
public:
    std::string name() const override { return "ntt-cpu-thread"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override;

    void config(const cli::CLI& cli) override;

private:
    using field = MontgomeryModInt<998244353>;

    static constexpr field one_ = field(1);
    static constexpr field zero_ = field(0);
    static constexpr u32 LIMIT = 1ULL << 20;

    std::vector<std::thread> threads;
    std::vector<field> root_bit_, iroot_bit_, carry_, icarry_;
    std::vector<field> roots_, iroots_;

    void set_root(u32 n);
    void dft(u32 n, field *a);
    void idft(u32 n, field *a);

    template<bool inverse>
    void bigBlock( u32 n, u32 block_size, field *a);

    template<bool inverse>
    void smallBlock( u32 n, u32 block_size, field *a);

    template<bool inverse>
    static void nttKernel(field *x_ptr, field *y_ptr, const field root, const u32 iter);

    static void nttKernelSimple(field *x_ptr, field *y_ptr, const u32 iter);

    u32 num_threads_ = std::thread::hardware_concurrency() ?
                        static_cast<u32>(std::thread::hardware_concurrency()) : 4;

};

BigInt NTT_cpu_thread::multiply(const BigInt &lhs, const BigInt &rhs) {
    const u32 lhs_len= lhs.size();
    const u32 rhs_len = rhs.size();

    if (lhs_len == 0 || rhs_len == 0) {
        return BigInt{};
    }

    threads.resize(num_threads_);

    const u32 result_len = std::max<u32>(4, lhs_len + rhs_len);
    set_root(result_len);

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

void NTT_cpu_thread::set_root(u32 n) {
    assert((n & (n - 1)) == 0);

    if (roots_.size() == n / 2) {
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
void NTT_cpu_thread::nttKernel(field *x_ptr, field *y_ptr, const field root, const u32 iter) {
    if constexpr (inverse) {
        for (u32 i = 0; i < iter; i++) {
            field x = *x_ptr;
            field y = *y_ptr;
            *x_ptr = x + y;
            *y_ptr = (x - y) * root;
            x_ptr++;
            y_ptr++;
        }
    } else {
        for (u32 i = 0; i < iter; i++) {
            field x = *x_ptr;
            field y = *y_ptr * root;
            *x_ptr = x + y;
            *y_ptr = (x - y);
            x_ptr++;
            y_ptr++;
        }
    }
}

void NTT_cpu_thread::nttKernelSimple(field *x_ptr, field *y_ptr, const u32 iter) {
    for (u32 i = 0; i < iter; i++) {
        field x = *x_ptr;
        field y = *y_ptr;
        *x_ptr = x + y;
        *y_ptr = (x - y);
        x_ptr++;
        y_ptr++;
    }
}

template<bool inverse>
void NTT_cpu_thread::bigBlock(
    u32 n, 
    u32 block_size, 
    field *a
) {
    const u32 half = block_size >> 1;
    const u32 block_num = n / block_size;
    const std::vector<field> &carry = inverse ? icarry_ : carry_;

    const u32 size_per_thread = (half + num_threads_ - 1) / num_threads_;

    // first block
    for (u32 i = 0; i < num_threads_; i++) {
        u32 start = i * size_per_thread;
        u32 end = std::min(start + size_per_thread, half);

        field *x_ptr = a + start;
        field *y_ptr = a + start + half;

        threads[i] = std::thread(nttKernelSimple, x_ptr, y_ptr, end - start);
    }
    for (u32 i = 0; i < num_threads_; i++) {
        threads[i].join();
    }

    // remain blocks
    field root = carry[__builtin_ctz(~0)];
    for (u32 block_id = 1; block_id < block_num; block_id++) {
        u32 start = block_id * block_size;
        u32 end = start + half;

        for (u32 i = 0; i < num_threads_; i++) {
            u32 l = start + i * size_per_thread;
            u32 r = std::min(end, l + size_per_thread);

            field *x_ptr = a + l;
            field *y_ptr = a + l + half;

            threads[i] = std::thread(nttKernel<inverse>, x_ptr, y_ptr, root, r - l);
        }

        for (u32 i = 0; i < num_threads_; i++) {
            threads[i].join();
        }

        root *= carry[__builtin_ctz(~block_id)];
    }
}

template<bool inverse> 
void NTT_cpu_thread::smallBlock(
    u32 n, 
    u32 block_size, 
    field *a
) {
    const u32 half = block_size >> 1;
    const u32 block_num = n / block_size;

    const std::vector<field> &roots = inverse ? iroots_ : roots_;
    const std::vector<field> &carry = inverse ? icarry_ : carry_;

    auto work = [&](u32 start_block, u32 end_block) -> void {
        field root = roots[start_block];

        for (u32 b = start_block; b < end_block; b++) {
            field *x_ptr = a + b * block_size;
            field *y_ptr = a + b * block_size + half;

            nttKernel<inverse>(x_ptr, y_ptr, root, half);

            root *= carry[__builtin_ctz(~b)];
        }
    };

    const u32 thread_num = std::min(block_num, num_threads_);
    const u32 block_per_thread = (block_num + thread_num - 1) / thread_num;

    for (u32 i = 0; i < thread_num; i++) {
        u32 start_block = i * block_per_thread;
        u32 end_block = std::min(block_num, (i + 1) * block_per_thread);
        threads[i] = std::thread(work, start_block, end_block);
    }

    for (u32 i = 0; i < thread_num; i++) {
        threads[i].join();
    }
}

void NTT_cpu_thread::dft(u32 n, field *a) {
    set_root(n);

    std::vector<std::thread> threads(num_threads_);

    for (u32 len = n; len >= 2; len >>= 1) {
        if (len >= LIMIT) {
            bigBlock<false>(n, len, a);
        } else {
            smallBlock<false>(n, len, a);
        }
    }

}

void NTT_cpu_thread::idft(u32 n, field *a) {
    set_root(n);

    for (u32 len = 2; len <= n; len <<= 1) {
        if (len >= LIMIT) {
            bigBlock<true>(n, len, a);
        } else {
            smallBlock<true>(n, len, a);
        }
    }

    for (u32 i = 0; i < n; i++) {
        a[i] /= n;
    }
}

void NTT_cpu_thread::config(const cli::CLI& cli) {
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
    register_impl("ntt-cpu-thread", [](){ return new NTT_cpu_thread(); });
    return true;
}();