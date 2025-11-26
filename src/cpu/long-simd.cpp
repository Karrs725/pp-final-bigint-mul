#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include <immintrin.h> 
class LongMulSimd : public BigMulImpl {
public:
    std::string name() const override { return "long-cpu-simd"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override {
        u32 len1 = lhs.size();
        u32 len2 = rhs.size();

        if (len1 == 0 || len2 == 0) {
            return BigInt{};
        }

        u32 result_len = len1 + len2 - 1;

        // 先把 BigInt 的 digit 搬到連續的 int32_t 陣列
        std::vector<int32_t> lhs32(len1);
        std::vector<int32_t> rhs32(len2);
        for (u32 i = 0; i < len1; ++i) {
            lhs32[i] = static_cast<int32_t>(lhs[i]);
        }
        for (u32 j = 0; j < len2; ++j) {
            rhs32[j] = static_cast<int32_t>(rhs[j]);
        }

        // 中間累加用 32-bit，避免直接對 BigInt 操作
        std::vector<int32_t> acc(result_len);
        std::fill(acc.begin(), acc.end(), 0);

        // schoolbook O(n^2)，外圈 i，內圈 j 用 SIMD
        for (u32 i = 0; i < len1; ++i) {
            int32_t ai = lhs32[i];
            if (ai == 0) continue;

            u32 j = 0;


            // 把 ai broadcast 成 8-lane 向量
            __m256i va = _mm256_set1_epi32(ai);

            // 一次處理 8 個 j： acc[i+j..i+j+7] += ai * rhs32[j..j+7]
            for (; j + 8 <= len2; j += 8) {
                // 讀 rhs 的 8 個 digit
                const __m256i vr   = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&rhs32[j])
                );
                // 讀 acc 的 8 個位置
                __m256i vacc = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&acc[i + j])
                );
                // 8 個乘法
                const __m256i prod = _mm256_mullo_epi32(va, vr);
                // 8 個加法
                vacc = _mm256_add_epi32(vacc, prod);
                // 寫回 acc
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(&acc[i + j]),
                    vacc
                );
            }

            // 尾巴不滿 8 的部份用 scalar 補
            for (; j < len2; ++j) {
                acc[i + j] += ai * rhs32[j];
            }
        }

        // 把 acc 做 base-10 的進位，寫回 BigInt
        BigInt result(result_len);
        int64_t carry = 0;
        for (u32 i = 0; i < result_len; ++i) {
            carry += acc[i];
            result[i] = static_cast<int>(carry % 10);
            carry /= 10;
        }
        while (carry) {
            result.push_back(static_cast<int>(carry % 10));
            carry /= 10;
        }

        // 跟原版一樣，看你 BigInt 有沒有 trim()
        // result.trim();

        return result;
    }
};

// 自動註冊成另一個實作
static bool _simd_longmul_reg = [](){
    register_impl("long-cpu-simd", [](){ return new LongMulSimd(); });
    return true;
}();

