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