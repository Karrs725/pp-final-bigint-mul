#include "bigint.hpp"

std::ostream &operator<<(std::ostream &os, const BigInt &big) {
    if (big.empty()) {
        os << "0";
        return os;
    }

    for (auto it = big.crbegin(); it != big.crend(); ++it) {
        os << *it;
    }

    return os;
}

BigInt::BigInt(std::string_view str_v) {
    auto valid = str_v 
        | std::views::drop_while( [](const char &c) { return c == '0'; } ) 
        | std::views::reverse;

    this->resize(valid.size());
    std::copy(valid.begin(), valid.end(), this->begin());

    for (i32 &val : (*this)) {
        assert(std::isdigit(val));
        val -= static_cast<i32>('0');
    }
}

BigInt random_bigint(u32 len) {
    assert(len > 0);

    static std::mt19937 rng(0x12345678);
    static std::uniform_int_distribution<int> dist(0, 9);

    BigInt result(len);
    for (u32 i = 0; i < len; ++i) {
        result[i] = dist(rng);
    }

    // Ensure the most significant digit is not zero
    if (result[len - 1] == 0) {
        result[len - 1] = (rand() % 9) + 1;
    }

    return result;
}