// LACS P0 kit -- graph helpers for the stand-alone sweep used by the E3
// baselines: compatibility mask (G_theta), core numbers, and k-core peeling of
// a vertex subset restricted to the query's component.

#include "p0_common.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" {

int p0_graph_version() { return 1; }

// mask[p] = 1 iff the CSR edge p = (v, idx[p]) satisfies
// d(x_v, x_u) <= min(theta_v, theta_u) * (1 + rel_eps)   (Lemma 3(b)).
int p0_gtheta_mask(int64_t n, const int64_t* ptr, const int32_t* idx, int m, const double* X,
                   const double* theta, double rel_eps, uint8_t* mask) {
    for (int64_t v = 0; v < n; ++v) {
        const double* xv = X + v * m;
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p) {
            const int64_t u = idx[p];
            const double t = std::min(theta[v], theta[u]) * (1.0 + rel_eps);
            mask[p] = p0::sqd(xv, X + u * m, m) <= t * t ? 1 : 0;
        }
    }
    return 0;
}

// Batagelj-Zaversnik core numbers of the subgraph formed by edges with
// mask[p] != 0 (mask == NULL: all edges).
int p0_core_numbers(int64_t n, const int64_t* ptr, const int32_t* idx, const uint8_t* mask,
                    int32_t* core) {
    std::vector<int32_t> deg((std::size_t)n);
    int32_t md = 0;
    for (int64_t v = 0; v < n; ++v) {
        int32_t d = 0;
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p)
            if (!mask || mask[p]) ++d;
        deg[(std::size_t)v] = d;
        if (d > md) md = d;
    }
    std::vector<int64_t> bin((std::size_t)md + 2, 0), pos((std::size_t)n), vert((std::size_t)n);
    for (int64_t v = 0; v < n; ++v) ++bin[(std::size_t)deg[(std::size_t)v]];
    int64_t start = 0;
    for (int32_t d = 0; d <= md; ++d) {
        const int64_t c = bin[(std::size_t)d];
        bin[(std::size_t)d] = start;
        start += c;
    }
    for (int64_t v = 0; v < n; ++v) {
        pos[(std::size_t)v] = bin[(std::size_t)deg[(std::size_t)v]];
        vert[(std::size_t)pos[(std::size_t)v]] = v;
        ++bin[(std::size_t)deg[(std::size_t)v]];
    }
    for (int32_t d = md; d >= 1; --d) bin[(std::size_t)d] = bin[(std::size_t)d - 1];
    bin[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t v = vert[(std::size_t)i];
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p) {
            if (mask && !mask[p]) continue;
            const int64_t u = idx[p];
            if (deg[(std::size_t)u] > deg[(std::size_t)v]) {
                const int32_t du = deg[(std::size_t)u];
                const int64_t pu = pos[(std::size_t)u];
                const int64_t pw = bin[(std::size_t)du];
                const int64_t w = vert[(std::size_t)pw];
                if (u != w) {
                    pos[(std::size_t)u] = pw;
                    vert[(std::size_t)pu] = w;
                    pos[(std::size_t)w] = pu;
                    vert[(std::size_t)pw] = u;
                }
                ++bin[(std::size_t)du];
                --deg[(std::size_t)u];
            }
        }
    }
    for (int64_t v = 0; v < n; ++v) core[(std::size_t)v] = deg[(std::size_t)v];
    return 0;
}

}  // extern "C"

namespace {

// k-core of the vertices with alive[v] = 1, then the component containing q.
// alive is modified in place (1 = kept). Returns the number of kept vertices
// (0 if q does not survive).
int peel_inplace(int n, const int64_t* ptr, const int32_t* idx, int q, int k, std::vector<uint8_t>& alive,
                 std::vector<int>& deg, std::vector<int>& stack) {
    stack.clear();
    // degrees first, with the initial mask; only then start removing (removing during
    // this pass would make later vertices miss the removed one and then be
    // decremented for it a second time)
    for (int v = 0; v < n; ++v) {
        if (!alive[(std::size_t)v]) continue;
        int d = 0;
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p)
            if (alive[(std::size_t)idx[p]]) ++d;
        deg[(std::size_t)v] = d;
    }
    for (int v = 0; v < n; ++v)
        if (alive[(std::size_t)v] && deg[(std::size_t)v] < k) {
            alive[(std::size_t)v] = 0;
            stack.push_back(v);
        }
    while (!stack.empty()) {
        const int v = stack.back();
        stack.pop_back();
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p) {
            const int u = idx[p];
            if (alive[(std::size_t)u] && --deg[(std::size_t)u] < k) {
                alive[(std::size_t)u] = 0;
                stack.push_back(u);
            }
        }
    }
    if (!alive[(std::size_t)q]) {
        std::fill(alive.begin(), alive.end(), 0);
        return 0;
    }
    std::vector<uint8_t> seen((std::size_t)n, 0);
    seen[(std::size_t)q] = 1;
    stack.push_back(q);
    int cnt = 0;
    while (!stack.empty()) {
        const int v = stack.back();
        stack.pop_back();
        ++cnt;
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p) {
            const int u = idx[p];
            if (alive[(std::size_t)u] && !seen[(std::size_t)u]) {
                seen[(std::size_t)u] = 1;
                stack.push_back(u);
            }
        }
    }
    for (int v = 0; v < n; ++v) alive[(std::size_t)v] = alive[(std::size_t)v] && seen[(std::size_t)v];
    return cnt;
}

}  // namespace

extern "C" {

// C_k(S, q): out_mask[v] = 1 for the component of q in the k-core of
// G[{v : in_mask[v]}] (local graph). Returns its size.
int p0_peel_component(int n, const int64_t* ptr, const int32_t* idx, const uint8_t* in_mask, int q, int k,
                      uint8_t* out_mask) {
    std::vector<uint8_t> alive(in_mask, in_mask + n);
    std::vector<int> deg((std::size_t)n), stack;
    if (q < 0 || q >= n || !alive[(std::size_t)q]) {
        std::fill(out_mask, out_mask + n, 0);
        return 0;
    }
    const int c = peel_inplace(n, ptr, idx, q, k, alive, deg, stack);
    for (int v = 0; v < n; ++v) out_mask[v] = alive[(std::size_t)v];
    return c;
}

// Greedy variant of the paper (Section 5.2): start from C_k(S, q); repeatedly
// delete the vertex other than q with the most conflicts (distance > tau),
// re-peel after each deletion, until no conflict remains. Ties: lowest index.
int p0_greedy_conflict(int n, const int64_t* ptr, const int32_t* idx, int m, const double* X, double tau,
                       int k, int q, const uint8_t* in_mask, double rel_eps, uint8_t* out_mask) {
    std::vector<uint8_t> alive(in_mask, in_mask + n);
    std::vector<int> deg((std::size_t)n), stack;
    std::fill(out_mask, out_mask + n, 0);
    if (q < 0 || q >= n || !alive[(std::size_t)q]) return 0;
    int cnt = peel_inplace(n, ptr, idx, q, k, alive, deg, stack);
    if (cnt == 0) return 0;
    const double t2 = (tau * (1.0 + rel_eps)) * (tau * (1.0 + rel_eps));
    std::vector<int> verts;
    for (int v = 0; v < n; ++v)
        if (alive[(std::size_t)v]) verts.push_back(v);
    // conflict lists among the initial survivors
    std::vector<std::vector<int>> conf((std::size_t)n);
    for (std::size_t a = 0; a < verts.size(); ++a)
        for (std::size_t b = a + 1; b < verts.size(); ++b) {
            const int u = verts[a], v = verts[b];
            if (p0::sqd(X + (int64_t)u * m, X + (int64_t)v * m, m) > t2) {
                conf[(std::size_t)u].push_back(v);
                conf[(std::size_t)v].push_back(u);
            }
        }
    std::vector<int> cc((std::size_t)n, 0);
    auto recount = [&]() {
        for (int v : verts) {
            if (!alive[(std::size_t)v]) continue;
            int c = 0;
            for (int u : conf[(std::size_t)v])
                if (alive[(std::size_t)u]) ++c;
            cc[(std::size_t)v] = c;
        }
    };
    recount();
    while (true) {
        int pick = -1, bestc = 0;
        for (int v : verts)
            if (alive[(std::size_t)v] && v != q && cc[(std::size_t)v] > bestc) {
                bestc = cc[(std::size_t)v];
                pick = v;
            }
        if (pick < 0) {
            // q itself may still conflict only if some partner conflicts with q,
            // which cannot happen for candidates within tau of q.
            break;
        }
        std::vector<uint8_t> before = alive;
        alive[(std::size_t)pick] = 0;
        cnt = peel_inplace(n, ptr, idx, q, k, alive, deg, stack);
        if (cnt == 0) return 0;
        // update conflict counts for every vertex that disappeared
        for (int v : verts)
            if (before[(std::size_t)v] && !alive[(std::size_t)v])
                for (int u : conf[(std::size_t)v])
                    if (alive[(std::size_t)u]) --cc[(std::size_t)u];
    }
    for (int v = 0; v < n; ++v) out_mask[v] = alive[(std::size_t)v];
    return cnt;
}

}  // extern "C"

extern "C" {

// Connected components of the subgraph induced by vertices with node_mask[v] = 1
// (all vertices if node_mask is NULL). comp[v] = -1 for excluded vertices.
// Returns the number of components.
int64_t p0_components(int64_t n, const int64_t* ptr, const int32_t* idx, const uint8_t* node_mask,
                      int32_t* comp) {
    for (int64_t v = 0; v < n; ++v) comp[v] = -1;
    std::vector<int64_t> stack;
    int64_t c = 0;
    for (int64_t s = 0; s < n; ++s) {
        if (comp[s] >= 0 || (node_mask && !node_mask[s])) continue;
        comp[s] = (int32_t)c;
        stack.clear();
        stack.push_back(s);
        while (!stack.empty()) {
            const int64_t v = stack.back();
            stack.pop_back();
            for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p) {
                const int64_t u = idx[p];
                if (comp[u] >= 0 || (node_mask && !node_mask[u])) continue;
                comp[u] = (int32_t)c;
                stack.push_back(u);
            }
        }
        ++c;
    }
    return c;
}

}  // extern "C"
