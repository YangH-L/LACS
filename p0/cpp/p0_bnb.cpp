// LACS P0 kit -- reference reimplementation of the paper's fixed-threshold
// branch and bound (Section 5.3, Lemma 5), so that every per-threshold solver
// runs inside the same sweep harness in E3. The paper's own engine remains the
// source of the reported Exact timings; this copy is for controlled comparisons
// and for cross-checking optimal sizes.
//
// Node (M, P): selected vertices M (q in M) and undecided P, no conflict inside
// M or between M and P. Rules, applied until nothing changes:
//   (1) edges between conflicting vertices are deleted once at the root; peel
//       G[M u P] to the component of q in its k-core; prune if a vertex of M goes;
//   (2) prune if |M u P| <= b; delete a vertex of P (prune if it is in M) that has
//       fewer than b conflict-free partners in M u P;
//   (3) greedy colouring of the compatibility graph on P (colour classes are
//       pairwise conflicting); prune if |M| + #colours <= b.
// If P has no conflict, M u P is optimal for the subtree. Otherwise branch on the
// vertex of P with most conflicts in P: delete it, or move it to M and delete its
// conflicting partners from P. b is the incumbent size.

#include "p0_bits.hpp"
#include "p0_common.hpp"

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

using p0::and_any;
using p0::and_count;
using p0::Bits;
using p0::for_each;
using p0::subset_of;

struct Bnb {
    int n = 0, k = 0, q = 0, branch_order = 0;
    std::vector<Bits> conf;
    std::vector<std::vector<int>> adj;
    int best_size = 0;
    std::vector<int> best;
    bool improved = false, timed_out = false;
    double time_limit = 0.0;
    p0::Timer timer;
    int64_t nodes = 0;
    double ub_root = -1;
    std::vector<int> deg, stack_;

    bool check_time() {
        if (time_limit > 0.0 && (nodes & 255) == 0 && timer.seconds() > time_limit) timed_out = true;
        return timed_out;
    }

    bool peel(const Bits& M, Bits& P) {
        Bits S = M;
        S.or_with(P);
        std::vector<int> verts;
        for_each(S, [&](int v) { verts.push_back(v); });
        for (int v : verts) {
            int d = 0;
            for (int u : adj[(std::size_t)v])
                if (S.test(u)) ++d;
            deg[(std::size_t)v] = d;
        }
        stack_.clear();
        for (int v : verts)
            if (deg[(std::size_t)v] < k) {
                S.reset(v);
                stack_.push_back(v);
            }
        while (!stack_.empty()) {
            const int v = stack_.back();
            stack_.pop_back();
            if (M.test(v)) return false;
            for (int u : adj[(std::size_t)v])
                if (S.test(u) && --deg[(std::size_t)u] < k) {
                    S.reset(u);
                    stack_.push_back(u);
                }
        }
        if (!S.test(q)) return false;
        Bits R(n);
        R.set(q);
        stack_.clear();
        stack_.push_back(q);
        while (!stack_.empty()) {
            const int v = stack_.back();
            stack_.pop_back();
            for (int u : adj[(std::size_t)v])
                if (S.test(u) && !R.test(u)) {
                    R.set(u);
                    stack_.push_back(u);
                }
        }
        if (!subset_of(M, R)) return false;
        P.and_with(R);
        return true;
    }

    bool reduce(const Bits& M, Bits& P) {
        while (true) {
            if (!peel(M, P)) return false;
            Bits S = M;
            S.or_with(P);
            const int s = S.count();
            if (s <= best_size) return false;
            bool changed = false;
            if (best_size > 0) {
                std::vector<int> drop;
                bool fail = false;
                for_each(S, [&](int v) {
                    if (fail) return;
                    const int cf = s - 1 - and_count(conf[(std::size_t)v], S);
                    if (cf < best_size) {
                        if (M.test(v))
                            fail = true;
                        else
                            drop.push_back(v);
                    }
                });
                if (fail) return false;
                for (int v : drop) P.reset(v);
                changed = !drop.empty();
            }
            if (!changed) return true;
        }
    }

    int colour_bound(const Bits& P) {
        std::vector<int> verts;
        for_each(P, [&](int v) { verts.push_back(v); });
        // compatibility degree within P, descending (Tomita-Seki style order)
        std::vector<std::pair<int, int>> ord;
        ord.reserve(verts.size());
        for (int v : verts) ord.push_back({and_count(conf[(std::size_t)v], P), v});
        std::sort(ord.begin(), ord.end());
        std::vector<Bits> classes;
        for (auto& pr : ord) {
            const int v = pr.second;
            bool placed = false;
            for (auto& cl : classes)
                if (subset_of(cl, conf[(std::size_t)v])) {
                    cl.set(v);
                    placed = true;
                    break;
                }
            if (!placed) {
                classes.emplace_back(n);
                classes.back().set(v);
            }
        }
        return (int)classes.size();
    }

    void search(Bits M, Bits P) {
        ++nodes;
        if (check_time()) return;
        if (!reduce(M, P)) return;
        bool conflict = false;
        int pivot = -1, best_c = -1;
        for_each(P, [&](int v) {
            const int c = and_count(conf[(std::size_t)v], P);
            if (c > 0) conflict = true;
            if (c > best_c) {
                best_c = c;
                pivot = v;
            }
        });
        if (!conflict) {
            Bits S = M;
            S.or_with(P);
            const int sz = S.count();
            if (sz > best_size) {
                best_size = sz;
                best.clear();
                for_each(S, [&](int v) { best.push_back(v); });
                improved = true;
            }
            return;
        }
        const int ub = M.count() + colour_bound(P);
        if (ub_root < 0) ub_root = ub;
        if (ub <= best_size) return;
        auto del = [&]() {
            Bits P2 = P;
            P2.reset(pivot);
            search(M, P2);
        };
        auto inc = [&]() {
            Bits M2 = M, P2 = P;
            M2.set(pivot);
            P2.reset(pivot);
            P2.andnot_with(conf[(std::size_t)pivot]);
            search(M2, P2);
        };
        if (branch_order == 0) {
            del();
            if (timed_out) return;
            inc();
        } else {
            inc();
            if (timed_out) return;
            del();
        }
    }
};

int bnb_run(int n, const int64_t* ptr, const int32_t* idx, int m, const double* X, double tau,
            const double* radius, int k, int q_local, int lb, double time_limit_s, int branch_order,
            double rel_eps, int32_t* out_members, int* out_size, double* out_stats);

}  // namespace

extern "C" {

int p0_bnb_version() { return 2; }

// Same interface and status codes as p0_krmax_solve.
int p0_bnb_solve(int n, const int64_t* ptr, const int32_t* idx, int m, const double* X, double tau,
                 int k, int q_local, int lb, double time_limit_s, int branch_order, double rel_eps,
                 int32_t* out_members, int* out_size, double* out_stats) {
    return bnb_run(n, ptr, idx, m, X, tau, nullptr, k, q_local, lb, time_limit_s, branch_order, rel_eps,
                   out_members, out_size, out_stats);
}

// Variant with per-vertex radii: u and v conflict iff d(u, v) > min(r_u, r_v).
// With r = theta this is the exact *pairwise* model of Section 6.3.
int p0_bnb_solve_radii(int n, const int64_t* ptr, const int32_t* idx, int m, const double* X,
                       const double* radius, int k, int q_local, int lb, double time_limit_s,
                       int branch_order, double rel_eps, int32_t* out_members, int* out_size,
                       double* out_stats) {
    return bnb_run(n, ptr, idx, m, X, 0.0, radius, k, q_local, lb, time_limit_s, branch_order, rel_eps,
                   out_members, out_size, out_stats);
}

}  // extern "C"

namespace {

int bnb_run(int n, const int64_t* ptr, const int32_t* idx, int m, const double* X, double tau,
            const double* radius, int k, int q_local, int lb, double time_limit_s, int branch_order,
            double rel_eps, int32_t* out_members, int* out_size, double* out_stats) {
    if (n <= 0 || q_local < 0 || q_local >= n || k < 1) return -1;
    Bnb b;
    b.n = n;
    b.k = k;
    b.q = q_local;
    b.branch_order = branch_order;
    b.best_size = lb > 0 ? lb : 0;
    b.time_limit = time_limit_s;
    b.conf.assign((std::size_t)n, Bits(n));
    b.adj.assign((std::size_t)n, {});
    b.deg.assign((std::size_t)n, 0);
    const double t2 = (tau * (1.0 + rel_eps)) * (tau * (1.0 + rel_eps));
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v) {
            double lim2 = t2;
            if (radius) {
                const double r = std::min(radius[u], radius[v]) * (1.0 + rel_eps);
                lim2 = r * r;
            }
            if (p0::sqd(X + (int64_t)u * m, X + (int64_t)v * m, m) > lim2) {
                b.conf[(std::size_t)u].set(v);
                b.conf[(std::size_t)v].set(u);
            }
        }
    for (int v = 0; v < n; ++v)
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p) {
            const int u = idx[p];
            if (u != v && !b.conf[(std::size_t)v].test(u)) b.adj[(std::size_t)v].push_back(u);
        }
    Bits M(n), P(n);
    M.set(q_local);
    for (int v = 0; v < n; ++v)
        if (v != q_local && !b.conf[(std::size_t)q_local].test(v)) P.set(v);
    {
        Bits P0 = P;
        out_stats[4] = b.peel(M, P0) ? 1 + P0.count() : 0;
    }
    b.search(M, P);
    out_stats[0] = (double)b.nodes;
    out_stats[1] = b.timer.seconds();
    out_stats[3] = b.ub_root;
    const int status = b.timed_out ? (b.improved ? 2 : 3) : (b.improved ? 0 : 1);
    out_stats[2] = status;
    if (b.improved) {
        std::sort(b.best.begin(), b.best.end());
        for (std::size_t i = 0; i < b.best.size(); ++i) out_members[i] = b.best[i];
        *out_size = (int)b.best.size();
    } else {
        *out_size = 0;
    }
    return 0;
}

}  // namespace
