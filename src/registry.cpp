#include "registry.hpp"

std::unordered_map<std::string, Creator>& impl_registry() {
    static std::unordered_map<std::string, Creator> registry;
    return registry;
}

void register_impl(const std::string& name, Creator c) {
    impl_registry()[name] = c;
}

BigMulImpl* get_impl(const std::string& name) {
    auto& registry = impl_registry();
    auto it = registry.find(name);
    if (it != registry.end()) {
        return it->second();
    }
    std::cerr << "Implementation not found: " << name << "\n";
    exit(1);
    // return nullptr;
}

std::vector<std::string> list_impls() {
    std::vector<std::string> names;
    for (const auto& pair : impl_registry()) {
        names.push_back(pair.first);
    }
    return names;
}