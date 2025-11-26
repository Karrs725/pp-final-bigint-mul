#pragma once

#include "common.hpp"

template <u32 mod>
class MontgomeryModInt {
public:
    using m32 = MontgomeryModInt;

    using value_type = u32;

    static __host__ __device__ constexpr u32 get_mod() {
        return mod;
    }

    static __host__ __device__ constexpr u32 get_primitive_root_prime() {
        u32 tmp[32]   = {};
        int cnt       = 0;
        const u32 phi = mod - 1;
        u32 m         = phi;

        for (u32 i = 2; i * i <= m; ++i) {
            if (m % i == 0) {
                tmp[cnt++] = i;

                do {
                    m /= i;
                } while (m % i == 0);
            }
        }

        if (m != 1)
            tmp[cnt++] = m;

        for (m32 res = 2;; res += 1) {
            bool f = true;

            for (int i = 0; i < cnt && f; ++i)
                f &= res.pow(phi / tmp[i]) != 1;

            if (f)
                return u32(res);
        }
    }

    __host__ __device__ constexpr MontgomeryModInt() = default;
    __host__ __device__ ~MontgomeryModInt()          = default;

    template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    __host__ __device__ constexpr MontgomeryModInt(T v) : v_(reduce(u64(v % i32(mod) + i32(mod)) * r2)) {}

    __host__ __device__ constexpr MontgomeryModInt(const m32 &) = default;

    __host__ __device__ constexpr u32 get() const {
        return norm(reduce(v_));
    }

    template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    __host__ __device__ explicit constexpr operator T() const {
        return T(get());
    }

    __host__ __device__ constexpr m32 operator-() const {
        m32 res;
        res.v_ = (mod2 & -(v_ != 0)) - v_;
        return res;
    }

    __host__ __device__ constexpr m32 inv() const {
        i32 x1 = 1, x3 = 0, a = get(), b = mod;

        while (b != 0) {
            i32 q = a / b, x1_old = x1, a_old = a;
            x1 = x3, x3 = x1_old - x3 * q, a = b, b = a_old - b * q;
        }

        return m32(x1);
    }

    __host__ __device__ constexpr m32 &operator=(const m32 &) = default;

    __host__ __device__ constexpr m32 &operator+=(const m32 &rhs) {
        v_ += rhs.v_ - mod2;
        v_ += mod2 & -(v_ >> 31);
        return *this;
    }
    __host__ __device__ constexpr m32 &operator-=(const m32 &rhs) {
        v_ -= rhs.v_;
        v_ += mod2 & -(v_ >> 31);
        return *this;
    }
    __host__ __device__ constexpr m32 &operator*=(const m32 &rhs) {
        v_ = reduce(u64(v_) * rhs.v_);
        return *this;
    }
    __host__ __device__ constexpr m32 &operator/=(const m32 &rhs) {
        return operator*=(rhs.inv());
    }
    friend __host__ __device__ constexpr m32 operator+(const m32 &lhs, const m32 &rhs) {
        return m32(lhs) += rhs;
    }
    friend __host__ __device__ constexpr m32 operator-(const m32 &lhs, const m32 &rhs) {
        return m32(lhs) -= rhs;
    }
    friend __host__ __device__ constexpr m32 operator*(const m32 &lhs, const m32 &rhs) {
        return m32(lhs) *= rhs;
    }
    friend __host__ __device__ constexpr m32 operator/(const m32 &lhs, const m32 &rhs) {
        return m32(lhs) /= rhs;
    }
    friend __host__ __device__ constexpr bool operator==(const m32 &lhs, const m32 &rhs) {
        return norm(lhs.v_) == norm(rhs.v_);
    }
    friend __host__ __device__ constexpr bool operator!=(const m32 &lhs, const m32 &rhs) {
        return norm(lhs.v_) != norm(rhs.v_);
    }

    friend std::istream &operator>>(std::istream &is, m32 &rhs) {
        i32 x;
        is >> x;
        rhs = m32(x);
        return is;
    }
    friend std::ostream &operator<<(std::ostream &os, const m32 &rhs) {
        return os << rhs.get();
    }

    __host__ __device__ constexpr m32 pow(u64 y) const {
        m32 res(1), x(*this);

        for (; y != 0; y >>= 1, x *= x)
            if (y & 1)
                res *= x;

        return res;
    }

private:
    static __host__ __device__ constexpr u32 get_r() {
        u32 two = 2, iv = mod * (two - mod * mod);
        iv *= two - mod * iv;
        iv *= two - mod * iv;
        return iv * (mod * iv - two);
    }

    static __host__ __device__ constexpr u32 reduce(u64 x) {
        return (x + u64(u32(x) * r) * mod) >> 32;
    }
    static __host__ __device__ constexpr u32 norm(u32 x) {
        return x - (mod & -((mod - 1 - x) >> 31));
    }

    u32 v_;

    static constexpr u32 r    = get_r();
    static constexpr u32 r2   = -u64(mod) % mod;
    static constexpr u32 mod2 = mod << 1;

    static_assert((mod & 1) == 1, "mod % 2 == 0\n");
    static_assert(-r * mod == 1, "???\n");
    static_assert((mod & (3U << 30)) == 0, "mod >= (1 << 30)\n");
    static_assert(mod != 1, "mod == 1\n");
};

template <u32 mod>
class ModInt {
public:
    using mi = ModInt;

    u32 v;

    constexpr ModInt() : v(0) {}
    template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    constexpr ModInt(T v_in) {
        i64 x = i64(v_in % i64(mod));
        if (x < 0) x += mod;
        v = u32(x);
    }

    static constexpr u32 get_mod() { return mod; }
    constexpr u32 get() const { return v; }


    static constexpr u32 get_primitive_root_prime() {
        static_assert(mod == 998244353, "Primitive root is only implemented for mod = 998244353");
        return 3;
    }

    constexpr mi operator-() const { return mi(0) - *this; }
    constexpr mi inv() const { return pow(mod - 2); }

    constexpr mi &operator+=(const mi &rhs) {
        v += rhs.v;
        if (v >= mod) v -= mod;
        return *this;
    }
    constexpr mi &operator-=(const mi &rhs) {
        v -= rhs.v;
        if (v >= mod) v += mod; // Handle underflow with unsigned types
        return *this;
    }
    constexpr mi &operator*=(const mi &rhs) {
        v = u64(v) * rhs.v % mod;
        return *this;
    }
    constexpr mi &operator/=(const mi &rhs) { return *this *= rhs.inv(); }

    friend constexpr mi operator+(const mi &lhs, const mi &rhs) { return mi(lhs) += rhs; }
    friend constexpr mi operator-(const mi &lhs, const mi &rhs) { return mi(lhs) -= rhs; }
    friend constexpr mi operator*(const mi &lhs, const mi &rhs) { return mi(lhs) *= rhs; }
    friend constexpr mi operator/(const mi &lhs, const mi &rhs) { return mi(lhs) /= rhs; }

    friend constexpr bool operator==(const mi &lhs, const mi &rhs) { return lhs.v == rhs.v; }
    friend constexpr bool operator!=(const mi &lhs, const mi &rhs) { return lhs.v != rhs.v; }

    constexpr mi pow(u64 y) const {
        mi res(1), x(*this);
        for (; y > 0; y >>= 1) {
            if (y & 1) res *= x;
            x *= x;
        }
        return res;
    }

    friend std::ostream &operator<<(std::ostream &os, const mi &rhs) {
        return os << rhs.get();
    }
    friend std::istream &operator>>(std::istream &is, mi &rhs) {
        i64 x;
        is >> x;
        rhs = mi(x);
        return is;
    }
};