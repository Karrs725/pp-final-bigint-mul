#pragma once

#include "common.hpp"

namespace cli {

struct option {
    std::string name;
    std::string argument;
    std::string description;
    bool requires_argument = false;
};

extern const std::vector<option> options;

// static const std::vector<option> options = {
//     {   .name = "help", 
//         .argument = "", 
//         .description = "Show this help message", 
//         .requires_argument = false 
//     },
//     {   .name = "list", 
//         .argument = "", 
//         .description = "List all available implementations", 
//         .requires_argument = false 
//     },
//     {   .name = "impl", 
//         .argument = "<name>", 
//         .description = "Run specific implementation", 
//         .requires_argument = true 
//     },
//     {   .name = "threads", 
//         .argument = "<n>", 
//         .description = "Number of threads to use", 
//         .requires_argument = true 
//     },
//     {   .name = "size", 
//         .argument = "<n>", 
//         .description = "Size of random numbers to generate", 
//         .requires_argument = true 
//     },
//     {   .name = "file", 
//         .argument = "<path>", 
//         .description = "Read input from file", 
//         .requires_argument = true 
//     }
// };

class CLI : public std::unordered_map<std::string, std::string> {
public:
    bool has_option(const std::string& name) const {
        return this->find(name) != this->end();
    }

    std::string get_option(const std::string& name, const std::string& default_value = "") const {
        auto it = this->find(name);
        if (it != this->end()) {
            return it->second;
        }
        return default_value;
    }
};

CLI parse_cli(int argc, char** argv);

void show_usage(const char* program_name);

}; // namespace cli