#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"

class SchoolbookThreadMul : public BigMulImpl {
private:
    // configured via CLI; if not set, we'll fall back to hardware_concurrency or env
    u32 num_threads_ = std::thread::hardware_concurrency() ?
                        static_cast<u32>(std::thread::hardware_concurrency()) : 4;

public:
    std::string name() const override { return "cpu-schoolbook-thread"; }

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

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) const override {
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

    static void partial_multiply(const BigInt& lhs, const BigInt& rhs, BigInt& partial_results, u32 start, u32 end) {
        u32 len2 = rhs.size();

        for (u32 i = start; i < end; i++) {
            for (u32 j = 0; j < len2; j++) {
                partial_results[i + j] += lhs[i] * rhs[j];
            }
        }
    }
};

// This object is automatically created when program loads
static bool _ = [](){
    register_impl("cpu-schoolbook-thread", [](){ return new SchoolbookThreadMul(); });
    return true;
}();