#include "bigint.hpp"

std::ostream &operator<<(std::ostream &os, const BigInt &big) {
    for (auto it = big.crbegin(); it != big.crend(); ++it) {
        os << *it;
    }
    return os;
}

BigInt::BigInt(const std::string &str) : BigInt(str.size()) {
    std::copy(str.crbegin(), str.crend(), this->begin());
    for (i32 &val : (*this)) {
        val -= static_cast<i32>('0');
    }
}

std::pair<BigInt, BigInt> read_bigint(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string str1, str2;
    std::getline(infile, str1);
    std::getline(infile, str2);

    BigInt num1(str1), num2(str2);

    infile.close();
    return {num1, num2};
}