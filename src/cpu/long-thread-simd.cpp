#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include <immintrin.h> 

class LongThreadMul_simd : public BigMulImpl {
private:
    // configured via CLI; if not set, we'll fall back to hardware_concurrency or env
    u32 num_threads_ = std::thread::hardware_concurrency() ?
                        static_cast<u32>(std::thread::hardware_concurrency()) : 4;

public:
    std::string name() const override { return "long-cpu-thread-simd"; }

    void config(const cli::CLI& cli) override {
        if (cli.has_option("threads")) {
            try {
                auto parsed = std::stoul(cli.get_option("threads"));
                if (parsed > 0) num_threads_ = static_cast<u32>(parsed);
            } catch (...) {
                // ignore invalid value
            }
        }
    }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override {
        u32 len1 = lhs.size();
        u32 len2 = rhs.size();
        u32 result_len = len1 + len2 - 1;
        BigInt result(result_len);
        
        // Prefer configured thread count; fall back to environment or default.
        u32 num_threads = num_threads_;
        std::cerr << "Using " << num_threads << " threads for multiplication.\n";

        std::vector<std::thread> threads(num_threads);
        std::vector<BigInt> partial_results(num_threads, BigInt(result_len));
        
        for (u32 i = 0; i < num_threads; i++) {
            threads[i] = std::thread(partial_multiply, std::ref(lhs), std::ref(rhs), std::ref(partial_results[i]),
            i * len1 / num_threads, (i + 1) * len1 / num_threads);
        }
        
        for (u32 i = 0; i < num_threads; i++) {
            threads[i].join();
        }
        
        for (u32 i = 0; i < num_threads; i++) {
            for (u32 j = 0; j < result_len; j++) {
                result[j] += partial_results[i][j];
            }
        }
        
        i32 carry = 0;
        for (u32 i = 0; i < result_len; i++)
        {
            result[i] += carry;
            carry = result[i] / 10;
            result[i] %= 10;
        }
        
        while (carry)
        {
            result.push_back(carry % 10);
            carry /= 10;
        }
        
        return result;
    }

   static void partial_multiply(const BigInt& lhs,
                             const BigInt& rhs,
                             BigInt& partial_results,
                             u32 start,
                             u32 end)
{
    const i32* lhs_ptr = lhs.data();            
    const i32* rhs_ptr = rhs.data();
    i32* out = partial_results.data();

    u32 len2 = rhs.size();

    for (u32 i = start; i < end; i++) {
        i32 a = lhs_ptr[i];
        if (a == 0) continue;

        // SIMD 廣播
        __m256i va = _mm256_set1_epi32(a);

        u32 j = 0;

        for (; j + 8 <= len2; j += 8) {
            __m256i vrhs = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(rhs_ptr + j)
            );

            __m256i vprod = _mm256_mullo_epi32(va, vrhs);

            __m256i vdst = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(out + (i + j))
            );

            vdst = _mm256_add_epi32(vdst, vprod);

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(out + (i + j)),
                vdst
            );
        }

        for (; j < len2; j++) {
            out[i + j] += a * rhs_ptr[j];
        }
    }
}

};

// This object is automatically created when program loads
static bool _ = [](){
    register_impl("long-cpu-thread-simd", [](){ return new LongThreadMul_simd(); });
    return true;
}();