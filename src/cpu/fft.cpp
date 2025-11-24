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

    // mod x^n - 1
    vector<field> fft(const vector<i32> &input, const u32 bit) {
        const int n = 1 << bit;
        assert(has_single_bit(n) && n >= input.size());

        vector<field> output(n);
        for (int i = 0; i < n; i++) {
            output[i] = (i < input.size()) ? field(input[i], 0) : field(0, 0);
        }

        for (int h = bit - 1; h >= 0; h--) {
            const u32 step = 1 << h;
            const field root = std::polar(1.0, numbers::pi / (double)step);
            for (u32 k = 0; k < n; k += step * 2) {
                field omega = field(1, 0);
                for (u32 j = 0; j < step; j++) {
                    field x = output[k + j] + output[k + j + step] * omega;
                    field y = output[k + j] - output[k + j + step] * omega;
                    output[k + j] = x;
                    output[k + j + step] = y;
                    omega *= root;
                }
            }
        }

        return output;
    }

    vector<i32> ifft(const vector<field> &input) {

    }

};

// This object is automatically created when program loads
static bool _ = [](){
    register_impl("cpu-fft", [](){ return new FFT_cpu(); });
    return true;
}();