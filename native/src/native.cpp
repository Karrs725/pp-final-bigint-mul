#include "bigint.hpp"
#include "multiply.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <bigint_file> <num_threads>" << std::endl;
        return 1;
    }

    u32 num_threads = 1;
    
    try {
        auto [num1, num2] = read_bigint(argv[1]);
        // std::cout << "First Big Integer: " << num1 << std::endl;
        // std::cout << "Second Big Integer: " << num2 << std::endl;
        if (argc == 3) {
            num_threads = std::stoi(argv[2]);
        }
        BigInt result = multiply(num1, num2, num_threads);
        std::cout << result << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}