#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"

#include "tatami/tatami.hpp"

#include <random>
#include <vector>
#include <chrono>
#include <memory>
#include <immintrin.h>

template<typename T, bool simd>
void run(int NR, int NC, int nthreads) {
    std::vector<T> contents(NR * NC);
    std::mt19937_64 rng(42);
    std::normal_distribution<T> dist;
    for (auto& x : contents) {
        x = dist(rng);
    }
    tatami::DenseRowMatrix<T, int> mat(NR, NC, std::move(contents));

    constexpr int alignment = 256;
    constexpr int jump = alignment / sizeof(T) / 8;

    std::vector<T> vec(NC + alignment); 
    auto vaptr = reinterpret_cast<void*>(vec.data());
    size_t space = vec.size();
    std::align(alignment, sizeof(T), vaptr, space);
    auto aptr = reinterpret_cast<T*>(vaptr);
    for (size_t c = 0; c < NC; ++c) {
        aptr[c] = dist(rng);
    }

    std::vector<T> output(nthreads);
    auto start = std::chrono::high_resolution_clock::now();

    tatami::parallelize([&](int t, int start, int len) -> void {
        auto wrk = mat.dense_row();
        std::vector<T> buffer(NC + alignment);
        auto vbptr = reinterpret_cast<void*>(buffer.data());
        size_t space = buffer.size();
        std::align(alignment, sizeof(T), vbptr, space);
        auto bptr = reinterpret_cast<T*>(vbptr);

        T total = 0;
        int upto = NC - NC % jump;

        for (int i = start, end = start + len; i < end; ++i) {
            auto xptr = wrk->fetch(i, bptr);
            tatami::copy_n(xptr, NC, bptr);

            if constexpr(!simd) {
                for (int j = 0; j < NC; ++j) {
                    auto x = xptr[j];
                    auto a = aptr[j];
                    bptr[j] = x / a + x * a;
                }

            } else {
                for (int j = 0; j < upto; j += jump) {
                    if constexpr(std::is_same<T, double>::value) {
                        auto x = _mm256_loadu_pd(xptr + j);
                        auto a = _mm256_loadu_pd(aptr + j);
                        _mm256_store_pd(bptr + j, _mm256_add_pd(_mm256_div_pd(x, a), _mm256_mul_pd(x, a)));
                    } else {
                        auto x = _mm256_loadu_ps(xptr + j);
                        auto a = _mm256_loadu_ps(aptr + j);
                        _mm256_store_ps(bptr + j, _mm256_add_ps(_mm256_div_ps(x, a), _mm256_mul_ps(x, a)));
                    }
                }

                for (int j = upto; j < NC; ++j) {
                    auto x = xptr[j];
                    auto a = aptr[j];
                    bptr[j] = x / a + x * a;
                }
            }

            // Just an operation to make sure that the results are used,
            // so the entire loop doesn't get optimized away.
            total += std::accumulate(bptr, bptr + NC, 0.0);
        }

        output[t] += total;
    }, NR, nthreads);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Access time: " << duration.count() << ", yielding " << std::accumulate(output.begin(), output.end(), 0.0) << std::endl;
}

int main(int argc, char** argv) {
    CLI::App app{"SIMD performance testing"};
    bool simd = false;
    app.add_flag("-s,--simd", simd, "Use SIMD");
    bool use_float = false;
    app.add_flag("-f,--float", use_float, "Use single-precision floats");
    int nthreads;
    app.add_option("-t,--threads", nthreads, "Number of threads")->default_val(1);
    int NR;
    app.add_option("-r,--nrow", NR, "Number of rows")->default_val(10000);
    int NC;
    app.add_option("-c,--ncol", NC, "Number of columns")->default_val(10000);
    CLI11_PARSE(app, argc, argv);

    if (use_float) {
        if (simd) {
            run<float, true>(NR, NC, nthreads);
        } else {
            run<float, false>(NR, NC, nthreads);
        }
    } else {
        if (simd) {
            run<double, true>(NR, NC, nthreads);
        } else {
            run<double, false>(NR, NC, nthreads);
        }
    }

    return 0;
}
