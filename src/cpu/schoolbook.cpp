#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"

class SchoolbookMul : public BigMulImpl {
public:
    std::string name() const override { return "cpu-schoolbook"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override {
        u32 len1 = lhs.size();
        u32 len2 = rhs.size();
        u32 result_len = len1 + len2 - 1;
        BigInt result(result_len);

        for (u32 i = 0; i < len1; i++) {
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
    register_impl("cpu-schoolbook", [](){ return new SchoolbookMul(); });
    return true;
}();

// BigInt multiply(const BigInt& lhs, const BigInt& rhs, u32 num_threads) {
//     u32 len1 = lhs.size();
//     u32 len2 = rhs.size(); 
//     u32 result_len = len1 + len2 - 1;
//     BigInt result(result_len);

//     std::vector<std::thread> threads(num_threads);
//     std::vector<BigInt> partial_results(num_threads, BigInt(result_len));

//     for (u32 i = 0; i < num_threads; i++) {
//         threads[i] = std::thread(partial_multiply, std::ref(lhs), std::ref(rhs), std::ref(partial_results[i]),
//                                  i * len1 / num_threads, (i + 1) * len1 / num_threads);
//     }

//     for (u32 i = 0; i < num_threads; i++) {
//         threads[i].join();
//     }

//     for (u32 i = 0; i < num_threads; i++) {
//         for (u32 j = 0; j < result_len; j++) {
//             result[j] += partial_results[i][j];
//         }
//     }

//     // #pragma omp parallel for num_threads(num_threads) reduction(+:temp_result)
//     // for (u32 i = 0; i < len1; i++) {
//     //     for (u32 j = 0; j < len2; j++) {
//     //         result[i + j] += lhs[i] * rhs[j];
//     //     }
//     // }

//     i32 carry = 0;
//     for (u32 i = 0; i < result_len; i++) {
//         result[i] += carry;
//         carry = result[i] / 10;
//         result[i] %= 10;
//     }

//     while (carry) {
//         result.push_back(carry % 10);
//         carry /= 10;
//     }

//     return result;
// }

// void partial_multiply(const BigInt& lhs, const BigInt& rhs, BigInt& partial_results, u32 start, u32 end) {
//     u32 len2 = rhs.size();

//     for (u32 i = start; i < end; i++) {
//         for (u32 j = 0; j < len2; j++) {
//             partial_results[i + j] += lhs[i] * rhs[j];
//         }
//     }
// }

// BigInt multiply(const BigInt& a, const BigInt& b, u32 num_threads);

// void partial_multiply(const BigInt& lhs, const BigInt& rhs, BigInt& partial_results, u32 start, u32 end);