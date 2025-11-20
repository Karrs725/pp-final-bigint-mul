#pragma once

#include "pch.hpp"

// struct BigInt {
//     std::basic_string<u32> value;

//     BigInt(const std::basic_string<u32>& val) : value(val) {}
//     u32 size() const {
//         return static_cast<u32>(value.size());
//     }

//     // Overload the output stream operator for easy printing
//     friend std::ostream& operator<<(std::ostream& os, const BigInt& bi) {
//         os << bi.value;
//         return os;
//     }

// };

struct BigInt : std::basic_string<i32> {
    BigInt() {}
    BigInt(u32 n) : std::basic_string<i32>(n, 0) {}
    BigInt(const std::string &str);
    friend std::ostream &operator<<(std::ostream &os, const BigInt &big);
};

std::pair<BigInt, BigInt> read_bigint(const std::string& filename);

// std::ostream &operator<<(std::ostream &os, const BigInt &big) {
//     for (auto it = big.crbegin(); it != big.crend(); ++it) {
//         os << *it;
//     }
//     return os;
// }