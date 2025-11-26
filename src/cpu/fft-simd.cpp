#include "bigint.hpp"
#include "common.hpp"
#include "registry.hpp"
#include <immintrin.h> 

class FFT_simd_cpu : public BigMulImpl {
public:
    std::string name() const override { return "fft-simd-cpu"; }

    BigInt multiply(const BigInt &lhs, const BigInt &rhs) override;

private:
    using fp =long double;                          // 改成 double，比 long double 更好向量化
    using field = std::complex<fp>;

    const field identity = field(1, 0);
    std::vector<field> roots, iroots, carry, icarry;

    void set_root(u32 n);
    void fft(u32 n, field *a);
    void ift(u32 n, field *a);
};

BigInt FFT_simd_cpu::multiply(const BigInt &lhs, const BigInt &rhs) {
    const u32 lhs_len= lhs.size();
    const u32 rhs_len = rhs.size();

    if (lhs_len == 0 || rhs_len == 0) {
        return BigInt{};
    }

    const u32 result_len = std::max<u32>(4, lhs_len + rhs_len);

    if (result_len >= u32(1E5)) {
        std::cerr << "Warning: floating-point FFT may lose precision for very large inputs (size >= 1E5).\n";
    }

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
    i32 carry_i = 0;
    for (u32 i = 0; i < result_len; i++) {
        carry_i += static_cast<i32>(std::round(A[i].real()));
        result[i] = carry_i % 10;
        carry_i /= 10;
    }
    assert(carry_i == 0);

    result.trim();

    return result;
}

void FFT_simd_cpu::set_root(u32 n) {
    assert((n & (n - 1)) == 0);

    const int h = std::__lg(n);
    roots.resize(h - 1);
    iroots.resize(h - 1);

    roots[h - 2]  = std::polar<fp>(1.0, 2 * std::numbers::pi_v<fp> / n);
    iroots[h - 2] = identity / roots[h - 2];

    for (int i = h - 3; i >= 0; i--) {
        roots[i]  = roots[i + 1] * roots[i + 1];
        iroots[i] = iroots[i + 1] * iroots[i + 1];
    }

    carry.resize(h - 1);
    icarry.resize(h - 1);
    field low = identity, ilow = identity;
    for (int i = 0; i < h - 1; i++) {
        carry[i]  = roots[i]  * ilow;
        icarry[i] = iroots[i] * low;
        low  *= roots[i];
        ilow *= iroots[i];
    }
}

// SIMD-friendly FFT：把 complex 乘加攤平成實部/虛部 + #pragma omp simd
void FFT_simd_cpu::fft(u32 n, field *a) {
     set_root(n);

    for (u32 len = n; len >= 2; len >>= 1) {
        u32 half = len >> 1;

        field root = identity;

        for (u32 i = 0, m = 0; i < n; i += len, m++) {
            // root (long double)
            fp wr_ld = root.real();
            fp wi_ld = root.imag();

            // 轉成 double，給 AVX2 用
            double wr = static_cast<double>(wr_ld);
            double wi = static_cast<double>(wi_ld);

            __m256d wr_vec = _mm256_set1_pd(wr);
            __m256d wi_vec = _mm256_set1_pd(wi);

            u32 j = 0;

            // 一次處理 4 個 butterfly（4 個 j）
            for (; j + 3 < half; j += 4) {
                // 把 4 個 x, y 實部/虛部抓出來轉為 double
                double xr_arr[4], xi_arr[4], yr_arr[4], yi_arr[4];

                for (int k = 0; k < 4; ++k) {
                    field x = a[i + j + k];
                    field y = a[i + j + k + half];

                    xr_arr[k] = static_cast<double>(x.real());
                    xi_arr[k] = static_cast<double>(x.imag());
                    yr_arr[k] = static_cast<double>(y.real());
                    yi_arr[k] = static_cast<double>(y.imag());
                }

                __m256d xr = _mm256_loadu_pd(xr_arr);
                __m256d xi = _mm256_loadu_pd(xi_arr);
                __m256d yr = _mm256_loadu_pd(yr_arr);
                __m256d yi = _mm256_loadu_pd(yi_arr);

                // t = y * root
                __m256d tr = _mm256_sub_pd(
                    _mm256_mul_pd(yr, wr_vec),
                    _mm256_mul_pd(yi, wi_vec)
                );
                __m256d ti = _mm256_add_pd(
                    _mm256_mul_pd(yr, wi_vec),
                    _mm256_mul_pd(yi, wr_vec)
                );

                // a0 = x + t
                // a1 = x - t
                __m256d out0r = _mm256_add_pd(xr, tr);
                __m256d out0i = _mm256_add_pd(xi, ti);
                __m256d out1r = _mm256_sub_pd(xr, tr);
                __m256d out1i = _mm256_sub_pd(xi, ti);

                double out0r_arr[4], out0i_arr[4], out1r_arr[4], out1i_arr[4];
                _mm256_storeu_pd(out0r_arr, out0r);
                _mm256_storeu_pd(out0i_arr, out0i);
                _mm256_storeu_pd(out1r_arr, out1r);
                _mm256_storeu_pd(out1i_arr, out1i);

                // 寫回 long double
                for (int k = 0; k < 4; ++k) {
                    a[i + j + k] =
                        field(static_cast<fp>(out0r_arr[k]),
                              static_cast<fp>(out0i_arr[k]));
                    a[i + j + k + half] =
                        field(static_cast<fp>(out1r_arr[k]),
                              static_cast<fp>(out1i_arr[k]));
                }
            }

            // 剩下尾巴（不足 4 個）用 scalar 處理
            for (; j < half; ++j) {
                field x = a[i + j];
                field y = a[i + j + half];

                fp xr = x.real();
                fp xi = x.imag();
                fp yr = y.real();
                fp yi = y.imag();

                fp tr = yr * wr_ld - yi * wi_ld;
                fp ti = yr * wi_ld + yi * wr_ld;

                a[i + j]        = field(xr + tr, xi + ti);
                a[i + j + half] = field(xr - tr, xi - ti);
            }

            root *= carry[__builtin_ctz(~m)];
        }
    }
}

void FFT_simd_cpu::ift(u32 n, field *a) {
     set_root(n);

    for (u32 len = 2; len <= n; len <<= 1) {
        u32 half = len >> 1;

        field iroot = identity;

        for (u32 i = 0, m = 0; i < n; i += len, m++) {
            fp wr_ld = iroot.real();
            fp wi_ld = iroot.imag();

            double wr = static_cast<double>(wr_ld);
            double wi = static_cast<double>(wi_ld);

            __m256d wr_vec = _mm256_set1_pd(wr);
            __m256d wi_vec = _mm256_set1_pd(wi);

            u32 j = 0;

            for (; j + 3 < half; j += 4) {
                double xr_arr[4], xi_arr[4], yr_arr[4], yi_arr[4];

                for (int k = 0; k < 4; ++k) {
                    field x = a[i + j + k];
                    field y = a[i + j + k + half];

                    xr_arr[k] = static_cast<double>(x.real());
                    xi_arr[k] = static_cast<double>(x.imag());
                    yr_arr[k] = static_cast<double>(y.real());
                    yi_arr[k] = static_cast<double>(y.imag());
                }

                __m256d xr = _mm256_loadu_pd(xr_arr);
                __m256d xi = _mm256_loadu_pd(xi_arr);
                __m256d yr = _mm256_loadu_pd(yr_arr);
                __m256d yi = _mm256_loadu_pd(yi_arr);

                // u = x + y
                __m256d ur = _mm256_add_pd(xr, yr);
                __m256d ui = _mm256_add_pd(xi, yi);

                // diff = x - y
                __m256d dr = _mm256_sub_pd(xr, yr);
                __m256d di = _mm256_sub_pd(xi, yi);

                // v = diff * iroot
                __m256d vr = _mm256_sub_pd(
                    _mm256_mul_pd(dr, wr_vec),
                    _mm256_mul_pd(di, wi_vec)
                );
                __m256d vi = _mm256_add_pd(
                    _mm256_mul_pd(dr, wi_vec),
                    _mm256_mul_pd(di, wr_vec)
                );

                double ur_arr[4], ui_arr[4], vr_arr[4], vi_arr[4];
                _mm256_storeu_pd(ur_arr, ur);
                _mm256_storeu_pd(ui_arr, ui);
                _mm256_storeu_pd(vr_arr, vr);
                _mm256_storeu_pd(vi_arr, vi);

                for (int k = 0; k < 4; ++k) {
                    a[i + j + k] =
                        field(static_cast<fp>(ur_arr[k]),
                              static_cast<fp>(ui_arr[k]));
                    a[i + j + k + half] =
                        field(static_cast<fp>(vr_arr[k]),
                              static_cast<fp>(vi_arr[k]));
                }
            }

            // scalar tail
            for (; j < half; ++j) {
                field x = a[i + j];
                field y = a[i + j + half];

                fp xr = x.real();
                fp xi = x.imag();
                fp yr = y.real();
                fp yi = y.imag();

                fp ur = xr + yr;
                fp ui = xi + yi;

                fp dr = xr - yr;
                fp di = xi - yi;

                fp vr = dr * wr_ld - di * wi_ld;
                fp vi = dr * wi_ld + di * wr_ld;

                a[i + j]        = field(ur, ui);
                a[i + j + half] = field(vr, vi);
            }

            iroot *= icarry[__builtin_ctz(~m)];
        }
    }

    fp inv_n = fp(1.0) / fp(n);

    // 這裡也可以寫 AVX2，但 scalar 已經很便宜了
    for (u32 i = 0; i < n; i++) {
        a[i] *= inv_n;
    }
}

// This object is automatically created when program loads
static bool _ = [](){
    register_impl("fft-simd-cpu", [](){ return new FFT_simd_cpu(); });
    return true;
}();
