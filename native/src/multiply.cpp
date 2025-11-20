#include "multiply.hpp"

BigInt multiply(const BigInt& lhs, const BigInt& rhs, u32 num_threads) {
    u32 len1 = lhs.size();
    u32 len2 = rhs.size(); 
    u32 result_len = len1 + len2 - 1;
    BigInt result(result_len);

    // #pragma omp parallel for num_threads(num_threads) reduction(+:temp_result)
    for (u32 i = 0; i < len1; i++) {
        for (u32 j = 0; j < len2; j++) {
            result[i + j] += lhs[i] * rhs[j];
        }
    }

    i32 carry = 0;
    for (u32 i = 0; i < result_len; i++) {
        result[i] += carry;
        carry = result[i] / 10;
        result[i] %= 10;
    }

    while (carry) {
        result.push_back(carry % 10);
        carry /= 10;
    }

    return result;
}