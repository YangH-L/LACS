// LACS P0 kit -- shared helpers for the C++ kernels.
// Build: see build.sh (g++ -O3 -march=native -std=c++17 -shared -fPIC).
#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstddef>

namespace p0 {

inline double sqd(const double* a, const double* b, int m) {
    double s = 0.0;
    for (int j = 0; j < m; ++j) {
        const double d = a[j] - b[j];
        s += d * d;
    }
    return s;
}

// xoshiro256** seeded with splitmix64.
struct Rng {
    uint64_t s[4];
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    explicit Rng(uint64_t seed) {
        uint64_t x = seed;
        for (int i = 0; i < 4; ++i) s[i] = splitmix(x);
    }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        const uint64_t result = rotl(s[1] * 5, 7) * 9;
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }
    // Uniform integer in [0, n).
    uint64_t below(uint64_t n) {
        return (uint64_t)(((__uint128_t)next() * (__uint128_t)n) >> 64);
    }
};

struct Dsu {
    std::vector<int> p, sz;
    explicit Dsu(int n) : p(n), sz(n, 1) {
        for (int i = 0; i < n; ++i) p[i] = i;
    }
    int find(int x) {
        while (p[x] != x) {
            p[x] = p[p[x]];
            x = p[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (sz[a] < sz[b]) { int t = a; a = b; b = t; }
        p[b] = a;
        sz[a] += sz[b];
    }
};

inline double diameter_sq(int n, int m, const double* X) {
    double best = 0.0;
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v) {
            const double d = sqd(X + (int64_t)u * m, X + (int64_t)v * m, m);
            if (d > best) best = d;
        }
    return best;
}

// Single-linkage clusters at squared radius eps2. Returns number of clusters,
// writes the size of the largest cluster to *largest.
inline int clusters(int n, int m, const double* X, double eps2, int* largest) {
    Dsu dsu(n);
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            if (sqd(X + (int64_t)u * m, X + (int64_t)v * m, m) <= eps2) dsu.unite(u, v);
    int cnt = 0, big = 0;
    for (int u = 0; u < n; ++u)
        if (dsu.find(u) == u) {
            ++cnt;
            if (dsu.sz[u] > big) big = dsu.sz[u];
        }
    *largest = big;
    return cnt;
}

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

}  // namespace p0
