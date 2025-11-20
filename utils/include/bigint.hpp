#include "pch.hpp"

struct BigInt {
    std::string value;

    BigInt(const std::string& val) : value(val) {}

    // Overload the output stream operator for easy printing
    friend std::ostream& operator<<(std::ostream& os, const BigInt& bi) {
        os << bi.value;
        return os;
    }

};

std::pair<BigInt, BigInt> read_bigint(const std::string& filename);