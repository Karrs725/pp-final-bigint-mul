#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"

class FFT_cpu : public BigMulImpl {
public:
    std::string name() const override { return "cpu-fft"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override {
        return rhs;
    }

private:
    using field = std::complex<double>;

    std::vector<field> w_, iw_;
    void set_root(u32 n) {
        w_.resize(n);
        iw_.resize(n);
        w_[0] = field(1, 0);
        iw_[0] = field(1, 0);
        field g = std::polar(1.0, 2 * std::numbers::pi / n);
        field ig = iw_[0] / std::polar(1.0, 2 * std::numbers::pi / n);
        for (u32 i = 1; i < n; i++) {
            w_[i] = w_[i - 1] * g;
            iw_[i] = iw_[i - 1] * ig;
        }
    }

    u32 bitrev(u32 x, u32 k) {
        u32 res = 0;
        for (u32 i = 0; i < k; i++) {
            res = (res << 1) | (x & 1);
            x >>= 1;
        }
        assert(x == 0);
        return res;
    }

    // mod x^n - 1
    void fft(u32 n, field *a) {
        set_root(n);
        for (u32 len = n, bit = 1; len >= 2; len >>= 1, bit++) {
            u32 half = len >> 1;
            for (u32 i = 0, m = 0; i < n; i += len, m++) {
                const field root = w_[bitrev(m, bit)];
                for (u32 j = 0; j < half; j++) {
                    field u = a[i + j];
                    field v = a[i + j + half] * root;
                    a[i + j] = u + v;
                    a[i + j + half] = u - v;
                }
            }
        }
    }


};

// This object is automatically created when program loads
static bool _ = [](){
    register_impl("cpu-fft", [](){ return new FFT_cpu(); });
    return true;
}();