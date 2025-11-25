#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"

class FFT_cpu : public BigMulImpl {
public:
    std::string name() const override { return "cpu-fft"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override {
        const u32 lhs_len= lhs.size();
        const u32 rhs_len = rhs.size();

        if (lhs_len == 0 || rhs_len == 0) {
            return BigInt{};
        }

        const u32 result_len = lhs_len + rhs_len;

        u32 fft_len = std::bit_ceil(result_len);

        std::vector<field> A(fft_len, field(0, 0));
        std::vector<field> B(fft_len, field(0, 0));

        for (u32 i = 0; i < fft_len; i++) {
            if (i < lhs_len) A[i] = field(lhs[i], 0);
            if (i < rhs_len) B[i] = field(rhs[i], 0);
        }

        fft(fft_len, A.data());
        fft(fft_len, B.data());
        for (u32 i = 0; i < fft_len; i++) {
            A[i] *= B[i];
        }
        ift(fft_len, A.data());

        BigInt result(result_len);
        i32 carry = 0;
        for (u32 i = 0; i < result_len; i++) {
            carry += static_cast<i32>(std::round(A[i].real()));
            result[i] = carry % 10;
            carry /= 10;
        }
        assert(carry == 0);

        result.trim();

        return result;
    }

private:
    using fp = long double;
    using field = std::complex<fp>;

    const field identity = field(1, 0);
    std::vector<field> roots, iroots, carry, icarry;

    void set_root(u32 n) {
        assert((n & (n - 1)) == 0);

        const int h = std::__lg(n);
        roots.resize(h - 1);
        iroots.resize(h - 1);

        roots[h - 2] = std::polar<fp>(1.0, 2 * std::numbers::pi / n);
        iroots[h - 2] = identity / roots[h - 2];

        for (int i = h - 3; i >= 0; i--) {
            roots[i] = roots[i + 1] * roots[i + 1];
            iroots[i] = iroots[i + 1] * iroots[i + 1];
        }

        carry.resize(h - 1);
        icarry.resize(h - 1);
        field low = identity, ilow = identity;
        for (int i = 0; i < h - 1; i++) {
            carry[i] = roots[i] * ilow;
            icarry[i] = iroots[i] * low;
            low *= roots[i];
            ilow *= iroots[i];
        }
    }

    void fft(u32 n, field *a) {
        set_root(n);

        for (u32 len = n; len >= 2; len >>= 1) {
            u32 half = len >> 1;
            field root = identity;
            for (u32 i = 0, m = 0; i < n; i += len, m++) {
                for (u32 j = 0; j < half; j++) {
                    field x = a[i + j];
                    field y = a[i + j + half] * root;
                    a[i + j] = x + y;
                    a[i + j + half] = x - y;
                }
                root *= carry[__builtin_ctz(~m)];
            }
        }
    }

    void ift(u32 n, field *a) {
        set_root(n);

        for (u32 len = 2; len <= n; len <<= 1) {
            u32 half = len >> 1;
            field iroot = identity;
            for (u32 i = 0, m = 0; i < n; i += len, m++) {
                for (u32 j = 0; j < half; j++) {
                    field x = a[i + j];
                    field y = a[i + j + half];
                    a[i + j] = x + y;
                    a[i + j + half] = (x - y) * iroot;
                }
                iroot *= icarry[__builtin_ctz(~m)];
            }
        }

        for (u32 i = 0; i < n; i++) {
            a[i] /= n;
        }
    }

};

// This object is automatically created when program loads
static bool _ = [](){
    register_impl("cpu-fft", [](){ return new FFT_cpu(); });
    return true;
}();