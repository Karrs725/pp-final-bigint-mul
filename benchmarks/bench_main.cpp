#include "common.hpp"
#include "bigint.hpp"
#include "registry.hpp"
#include "cli.hpp"

class Benchmark {
private:
    const static std::string ref_impl_name;
    BigInt num1, num2, ans;
    bool show = false;

public:
    Benchmark() = default;
    Benchmark(const cli::CLI cli) {
        setup(cli);
    }

    void setup(const cli::CLI cli) {
        if (cli.has_option("file")) {
            load(cli.get_option("file"));
        } else {
            u32 size = 1000; // default size
            if (cli.has_option("size")) {
                size = static_cast<u32>(std::stoul(cli.get_option("size")));
            }
            random(size);
        }

        if (cli.has_option("show")) {
            show = true;
        }

    }

    void random(u32 size) {
        num1 = random_bigint(size);
        num2 = random_bigint(size);
        auto ref_impl = get_impl(ref_impl_name);
        if (!ref_impl) {
            throw std::runtime_error("Reference implementation not found");
        }
        ans = ref_impl->multiply(num1, num2);
    }

    void load(const std::string &filename) {
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

    bool run_bench(BigMulImpl *impl) {

        auto start = std::chrono::steady_clock::now();
        BigInt result = impl->multiply(num1, num2);
        auto end = std::chrono::steady_clock::now();

        auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        bool correct = (result == ans);

        std::cout << "-------------------------------------------\n";
        std::cout << "Benchmarking   : " << impl->name() << "\n";
        std::cout << "Input size     : " << num1.size() << " x " << num2.size() << "\n";
        std::cout << "Time taken     : " << time_ms << " ms\n";
        std::cout << "Result correct : " << (correct ? "YES" : "NO") << "\n";
        if (show) {
        std::cout << "Num1           : " << num1 << "\n";
        std::cout << "Num2           : " << num2 << "\n";
        std::cout << "Result         : " << result << "\n";
        }
        std::cout << "-------------------------------------------\n";
        return correct;
    }
};

const std::string Benchmark::ref_impl_name = "schoolbook-cpu";

int main(int argc, char** argv) {

    using namespace cli;

    // Show help if no arguments
    if (argc == 1) {
        show_usage(argv[0]);
        return 0;
    }

    CLI cli = parse_cli(argc, argv);

    // Show help
    if (cli.has_option("help")) {
        show_usage(argv[0]);
        return 0;
    }

    // List implementations
    if (cli.has_option("list")) {
        std::cout << "Available implementations:\n";
        for (auto& name : list_impls()) {
            std::cout << "  • " << name << "\n";
        }
        return 0;
    }

    Benchmark bench(cli);

    // If no impl is specified, run all
    std::vector<BigMulImpl*> impls_to_run;
    if (cli.has_option("impl")) {
        const std::string impl_name = cli.get_option("impl");

        BigMulImpl* impl = get_impl(impl_name);
        impl->config(cli);
        impls_to_run.push_back(impl);
    } else {
        for (const auto& name : list_impls()) {
            BigMulImpl* impl = get_impl(name);
            impl->config(cli);
            impls_to_run.push_back(impl);
        }
    }

    for (auto impl : impls_to_run) {
        bench.run_bench(impl);
    }

    for (auto impl : impls_to_run) {
        delete impl;
    }

    return 0;
}
