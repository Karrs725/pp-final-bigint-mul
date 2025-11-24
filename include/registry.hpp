#pragma once

#include "common.hpp"
#include "bigint.hpp"
#include "cli.hpp"

struct BigMulImpl {
    virtual std::string name() const = 0;

    virtual void config(const cli::CLI&) {};

    virtual BigInt multiply(const BigInt& lhs, const BigInt& rhs) const = 0;

    virtual ~BigMulImpl() {}
};

using Creator = std::function<BigMulImpl*()>;

std::unordered_map<std::string, Creator>& impl_registry();
void register_impl(const std::string& name, Creator c);

BigMulImpl* get_impl(const std::string& name);
std::vector<std::string> list_impls();