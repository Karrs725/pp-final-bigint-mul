#include "bigint.hpp"

std::pair<BigInt, BigInt> read_bigint(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string num1, num2;
    std::getline(infile, num1);
    std::getline(infile, num2);

    infile.close();
    return {num1, num2};
}