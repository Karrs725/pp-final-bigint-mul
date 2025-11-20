#include "bigint.hpp"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <bigint_file>" << std::endl;
        return 1;
    }

    try {
        auto [num1, num2] = read_bigint(argv[1]);
        std::cout << "First Big Integer: " << num1 << std::endl;
        std::cout << "Second Big Integer: " << num2 << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}