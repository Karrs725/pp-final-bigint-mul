#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"

class LongOmpMul : public BigMulImpl {
public:
    std::string name() const override { return "long-cpu-omp"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override {
        u32 len1 = lhs.size();
        u32 len2 = rhs.size();
        u32 result_len = len1 + len2 - 1;
        BigInt result(result_len);

        for (u32 i = 0; i < len1; i++) {
            #pragma omp parallel for
            for (u32 j = 0; j < len2; j++) {
                result[i + j] += lhs[i] * rhs[j];
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
};

// This object is automatically created when program loads
static bool _ = [](){
    register_impl("long-cpu-omp", [](){ return new LongOmpMul(); });
    return true;
}();