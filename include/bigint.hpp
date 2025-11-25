#pragma once

#include "common.hpp"

struct BigInt : std::basic_string<i32> {
    BigInt() {}
    BigInt(u32 n) : std::basic_string<i32>(n, 0) {}
    BigInt(std::string_view str_v);
    void trim() {
        while (!this->empty() && this->back() == 0) {
            this->pop_back();
        }
    }
    friend std::ostream &operator<<(std::ostream &os, const BigInt &big);
};

BigInt random_bigint(u32 len);