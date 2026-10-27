// LACS P0 kit -- small dynamic bitset used by the branch-and-bound kernels.
#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

namespace p0 {

using Word = uint64_t;

struct Bits {
    std::vector<Word> w;
    Bits() {}
    explicit Bits(int n) : w((std::size_t)((n + 63) >> 6), 0) {}
    void set(int i) { w[(std::size_t)(i >> 6)] |= (Word)1 << (i & 63); }
    void reset(int i) { w[(std::size_t)(i >> 6)] &= ~((Word)1 << (i & 63)); }
    bool test(int i) const { return (w[(std::size_t)(i >> 6)] >> (i & 63)) & 1; }
    int count() const {
        int c = 0;
        for (Word x : w) c += __builtin_popcountll(x);
        return c;
    }
    bool any() const {
        for (Word x : w)
            if (x) return true;
        return false;
    }
    void or_with(const Bits& o) {
        for (std::size_t i = 0; i < w.size(); ++i) w[i] |= o.w[i];
    }
    void and_with(const Bits& o) {
        for (std::size_t i = 0; i < w.size(); ++i) w[i] &= o.w[i];
    }
    void andnot_with(const Bits& o) {
        for (std::size_t i = 0; i < w.size(); ++i) w[i] &= ~o.w[i];
    }
};

inline int and_count(const Bits& a, const Bits& b) {
    int c = 0;
    for (std::size_t i = 0; i < a.w.size(); ++i) c += __builtin_popcountll(a.w[i] & b.w[i]);
    return c;
}
inline bool and_any(const Bits& a, const Bits& b) {
    for (std::size_t i = 0; i < a.w.size(); ++i)
        if (a.w[i] & b.w[i]) return true;
    return false;
}
// true iff a is a subset of b
inline bool subset_of(const Bits& a, const Bits& b) {
    for (std::size_t i = 0; i < a.w.size(); ++i)
        if (a.w[i] & ~b.w[i]) return false;
    return true;
}
template <class F>
inline void for_each(const Bits& a, F f) {
    for (std::size_t i = 0; i < a.w.size(); ++i) {
        Word x = a.w[i];
        while (x) {
            const int b = __builtin_ctzll(x);
            f((int)(i * 64 + (std::size_t)b));
            x &= x - 1;
        }
    }
}

}  // namespace p0
