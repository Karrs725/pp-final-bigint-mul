#include "common.hpp"
#include "bigint.hpp"
#include "registry.hpp"

class Benchmark {
private:
    const static std::string ref_impl_name;
    BigInt num1, num2, ans;

public:
    Benchmark() = default;
    void setup(u32 size) {
        num1 = random_bigint(size);
        num2 = random_bigint(size);
        auto ref_impl = get_impl(ref_impl_name);
        if (!ref_impl) {
            throw std::runtime_error("Reference implementation not found");
        }
        ans = ref_impl->multiply(num1, num2);
    }

    void setup(const std::string &filename) {
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            throw std::runtime_error("Could not open benchmark: " + filename);
        }

        std::string line;
        for (BigInt *ptr : {&num1, &num2, &ans}) {
            std::getline(infile, line);
            *ptr = BigInt(line);
        }
        infile.close();
    }

    bool run_bench(const std::string& impl_name) {
        auto impl = get_impl(impl_name);
        if (!impl) {
            std::cerr << "Implementation not found: " << impl_name << "\n";
            return false;
        }

        auto start = std::chrono::steady_clock::now();

        BigInt result = impl->multiply(num1, num2);

        auto end = std::chrono::steady_clock::now();
        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        bool correct = (result == ans);

        std::cout << "-------------------------------------------\n";
        std::cout << "Benchmarking   : " << impl_name << "\n";
        std::cout << "Input size     : " << num1.size() << " x " << num2.size() << "\n";
        std::cout << "Time taken     : " << time_ms << " ms\n";
        std::cout << "Result correct : " << (correct ? "YES" : "NO") << "\n";
        std::cout << "-------------------------------------------\n";
        return correct;
    }
};

const std::string Benchmark::ref_impl_name = "cpu-schoolbook";

struct CLI {
    std::string impl = "";
    u32 size = 1000;
    bool list = false;
    bool help = false;
    std::string file = "";
};

// -----------------------------------------------------------
// Show usage information
// -----------------------------------------------------------
void show_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Benchmark tool for large integer multiplication algorithms\n\n";
    std::cout << "OPTIONS:\n";
    std::cout << "  --help               Show this help message\n";
    std::cout << "  --list               List all available implementations\n";
    std::cout << "  --impl <name>        Run specific implementation (default: all)\n";
    std::cout << "  --size <n>           Size of random numbers to generate (default: 1000)\n";
    std::cout << "  --file <path>        Read input from file (format: two numbers on separate lines)\n";
    std::cout << "\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  " << program_name << " --list\n";
    std::cout << "  " << program_name << " --size 500\n";
    std::cout << "  " << program_name << " --impl cpu-schoolbook --size 1000\n";
    std::cout << "  " << program_name << " --file benchmarks/random/nd3_1.in\n";
    std::cout << "\n";
}

// -----------------------------------------------------------
// Simple CLI parser
// -----------------------------------------------------------
CLI parse_cli(int argc, char** argv) {
    CLI cli;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            cli.help = true;
        }
        else if (arg == "--impl" && i + 1 < argc) {
            cli.impl = argv[++i];
        }
        else if (arg == "--size" && i + 1 < argc) {
            cli.size = std::stoull(argv[++i]);
        }
        else if (arg == "--file" && i + 1 < argc) {
            cli.file = argv[++i];
        }
        else if (arg == "--list") {
            cli.list = true;
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Use --help for usage information\n";
            exit(1);
        }
    }
    return cli;
}

// -----------------------------------------------------------
// Main
// -----------------------------------------------------------
int main(int argc, char** argv) {
    // Show help if no arguments
    if (argc == 1) {
        show_usage(argv[0]);
        return 0;
    }

    CLI cli = parse_cli(argc, argv);

    // Show help
    if (cli.help) {
        show_usage(argv[0]);
        return 0;
    }

    // List implementations
    if (cli.list) {
        std::cout << "Available implementations:\n";
        for (auto& name : list_impls()) {
            std::cout << "  • " << name << "\n";
        }
        return 0;
    }

    Benchmark bench;
    if (!cli.file.empty()) {
        bench.setup(cli.file);
    } else {
        bench.setup(cli.size);
    }

    // If no impl is specified, run all
    if (cli.impl.empty()) {
        std::cout << "\nRunning all implementations:\n\n";

        for (const auto& name : list_impls()) {
            bench.run_bench(name);
        }
    } else {
        // Run only the one requested
        std::cout << "\nRunning implementation: " << cli.impl << "\n\n";
        bench.run_bench(cli.impl);
    }

    return 0;
}
