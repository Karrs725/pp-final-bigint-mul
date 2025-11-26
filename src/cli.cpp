#include "common.hpp"
#include "cli.hpp"

namespace cli {

const std::vector<option> options = {
    {   .name = "help",
        .argument = "",
        .description = "Show this help message",
        .requires_argument = false
    },
    {   .name = "list",
        .argument = "",
        .description = "List all available implementations",
        .requires_argument = false
    },
    {   .name = "impl",
        .argument = "<name>",
        .description = "Run specific implementation",
        .requires_argument = true
    },
    {   .name = "threads",
        .argument = "<n>",
        .description = "Number of threads to use",
        .requires_argument = true
    },
    {   .name = "size",
        .argument = "<n>",
        .description = "Size of random numbers to generate",
        .requires_argument = true
    },
    {   .name = "file",
        .argument = "<path>",
        .description = "Read input from file",
        .requires_argument = true
    },
    {   .name = "show",
        .argument = "",
        .description = "Display input and output numbers",
        .requires_argument = false
    },
    {   .name = "iter",
        .argument = "<n>",
        .description = "Number of iterations for benchmarking",
        .requires_argument = true
    }
};

CLI parse_cli(int argc, char** argv) {
    CLI cli;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        bool matched = false;
        for (const auto& opt : options) {
            if (arg == "--" + opt.name) {
                if (opt.requires_argument) {
                    if (i + 1 < argc) {
                        cli[opt.name] = argv[++i];
                    } else {
                        std::cerr << "Error: Option --" << opt.name << " requires an argument.\n";
                        exit(1);
                    }
                } else {
                    cli[opt.name] = "true";
                }
                matched = true;
                break;
            }
        }

        if (matched) continue;

        std::cerr << "Unknown argument: " << arg << "\n";
        std::cerr << "Use --help for usage information\n";
        exit(1);
    }

    return cli;
}

void show_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Benchmark tool for large integer multiplication algorithms\n\n";
    std::cout << "OPTIONS:\n";
    for (const auto& opt : options) {
        std::cout << std::left << std::setw(12) << "  --" + opt.name;
        std::cout << std::left << std::setw(8);
        std::cout << (opt.requires_argument ? opt.argument : "");
        std::cout << std::left << std::setw(10) << opt.description << "\n";
    }
    std::cout << "\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  " << program_name << " --list\n";
    std::cout << "  " << program_name << " --size 500\n";
    std::cout << "  " << program_name << " --impl cpu-long --size 1000\n";
    std::cout << "  " << program_name << " --file benchmarks/random/nd3_1.in\n";
    std::cout << "\n";
}

} // namespace cli
