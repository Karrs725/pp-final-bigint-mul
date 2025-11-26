#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include <omp.h>

class LongOmpMul : public BigMulImpl {
public:
    std::string name() const override { return "long-cpu-omp"; }

    void config(const cli::CLI& cli) override {
        if (cli.has_option("threads")) {
            try {
                auto parsed = std::stoul(cli.get_option("threads"));
                if (parsed > 0) num_threads_ = static_cast<u32>(parsed);
            } catch (...) {}
        }
    }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override {
        const u32 len1 = lhs.size();
        const u32 len2 = rhs.size();
        
        if (len1 == 0 || len2 == 0) return BigInt{};
        
        const u32 result_len = len1 + len2;
        
        // Use i64 to accumulate products without overflow concerns
        // Each digit is 0-9, so max product per cell is 9*9 = 81
        // With n additions, max value is 81*n which fits in i64
        std::vector<i64> accum(result_len, 0);

        omp_set_num_threads(num_threads_);

        // Strategy: parallelize over diagonal bands to avoid race conditions
        // Each diagonal d contains elements where i + j = d
        // All elements on the same diagonal can be computed independently
        
        // Alternative: parallelize over blocks of the output array
        // Each thread computes a range of output positions
        
        #pragma omp parallel
        {
            // Thread-local accumulator to avoid false sharing
            std::vector<i64> local_accum(result_len, 0);
            
            // Each thread processes a subset of rows
            #pragma omp for schedule(static)
            for (u32 i = 0; i < len1; i++) {
                i64 a = lhs[i];
                for (u32 j = 0; j < len2; j++) {
                    local_accum[i + j] += a * rhs[j];
                }
            }
            
            // Reduce local accumulators into global accumulator
            #pragma omp critical
            {
                for (u32 k = 0; k < result_len; k++) {
                    accum[k] += local_accum[k];
                }
            }
        }

        // Sequential carry propagation
        BigInt result(result_len);
        i64 carry = 0;
        for (u32 i = 0; i < result_len; i++) {
            i64 sum = accum[i] + carry;
            result[i] = sum % 10;
            carry = sum / 10;
        }

        // Handle remaining carry
        while (carry > 0) {
            result.push_back(carry % 10);
            carry /= 10;
        }

        result.trim();
        return result;
    }

private:
    u32 num_threads_ = std::thread::hardware_concurrency() ?
                        static_cast<u32>(std::thread::hardware_concurrency()) : 4;
};

// This object is automatically created when program loads
static bool _ = [](){
    register_impl("long-cpu-omp", [](){ return new LongOmpMul(); });
    return true;
}();