// LACS P0 kit -- per-threshold exact baseline: maximum (k,r)-core containing q,
// reimplementing AdvMax of Zhang et al. (PVLDB 2017; VLDB J. 2019), reference [35].
//
// Input is one fixed-threshold subproblem of LatSweep (Lemma 4): a candidate set
// with q, the structural graph on it, positions, and tau. Two candidates
// "conflict" (are dissimilar) iff their distance exceeds tau. We search for a
// largest set H with q in H, G[H] connected, min degree >= k, and no conflict in
// H, of size strictly larger than lb (the incumbent).
//
// Elements of [35] that are kept:
//   * preprocessing: delete structural edges between dissimilar endpoints,
//     then k-core;
//   * search state (M, C, E): chosen, candidate and excluded vertices; E keeps
//     excluded vertices that are similar to all of M;
//   * structure pruning (k-core of M u C) and similarity pruning (drop
//     candidates dissimilar to a chosen vertex);
//   * retaining candidates: vertices of SF(C) (similar to all of C) are never
//     branched on; if C = SF(C), M u C is a solution;
//   * direct move (Remark 1): u in SF(C) with >= k neighbours in M joins M;
//   * early termination (Theorem 5, conditions (i) and (ii));
//   * the (k,k')-core upper bound (Algorithm 6);
//   * search order lambda*Delta1 - Delta2 with lambda = 5 (two-hop estimates),
//     the branch with the higher score explored first.
// Adaptations to the query version (all sound):
//   * q is fixed in M from the root, and after structure pruning only the
//     connected component containing q is kept;
//   * the (k,k')-core bound is read at q: the level at which q is peeled
//     (a solution containing q of size s is a (k, s-1)-core containing q);
//   * early-termination condition (ii) only uses excluded vertices that are
//     connected to M through excluded vertices, so the extended set stays
//     connected.
// order_mode 1 replaces the [35] order by "most conflicts in C, expand first".

#include "p0_bits.hpp"
#include "p0_common.hpp"

#include <algorithm>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using p0::and_any;
using p0::and_count;
using p0::Bits;
using p0::for_each;

struct Solver {
    int n = 0, k = 0, q = 0, order_mode = 0;
    double lambda = 5.0;
    std::vector<Bits> conf;              // dissimilar pairs
    std::vector<std::vector<int>> adj;   // structural edges between similar endpoints
    int best_size = 0;
    std::vector<int> best;
    bool improved = false;
    bool timed_out = false;
    double time_limit = 0.0;
    p0::Timer timer;
    int64_t nodes = 0;
    double ub_root = -1;
    // scratch
    std::vector<int> deg_g, deg_s, stack_, order_;
    std::vector<uint8_t> alive;

    bool check_time() {
        if (time_limit > 0.0 && (nodes & 255) == 0 && timer.seconds() > time_limit) timed_out = true;
        return timed_out;
    }

    // k-core of G[M u C] restricted to q's component. Updates C in place.
    // Returns false if a vertex of M is removed (branch infeasible).
    bool peel(const Bits& M, Bits& C) {
        Bits S = M;
        for (std::size_t i = 0; i < S.w.size(); ++i) S.w[i] |= C.w[i];
        std::vector<int> verts;
        for_each(S, [&](int v) { verts.push_back(v); });
        for (int v : verts) {
            int d = 0;
            for (int u : adj[v])
                if (S.test(u)) ++d;
            deg_g[v] = d;
        }
        stack_.clear();
        for (int v : verts)
            if (deg_g[v] < k) {
                stack_.push_back(v);
                S.reset(v);
            }
        while (!stack_.empty()) {
            const int v = stack_.back();
            stack_.pop_back();
            if (M.test(v)) return false;
            for (int u : adj[v])
                if (S.test(u)) {
                    if (--deg_g[u] < k) {
                        S.reset(u);
                        stack_.push_back(u);
                    }
                }
        }
        if (!S.test(q)) return false;
        // q's component
        Bits R(n);
        stack_.clear();
        stack_.push_back(q);
        R.set(q);
        while (!stack_.empty()) {
            const int v = stack_.back();
            stack_.pop_back();
            for (int u : adj[v])
                if (S.test(u) && !R.test(u)) {
                    R.set(u);
                    stack_.push_back(u);
                }
        }
        // M must be inside R
        for (std::size_t i = 0; i < M.w.size(); ++i)
            if (M.w[i] & ~R.w[i]) return false;
        for (std::size_t i = 0; i < C.w.size(); ++i) C.w[i] &= R.w[i];
        return true;
    }

    int deg_in(int v, const Bits& S) const {
        int d = 0;
        for (int u : adj[v])
            if (S.test(u)) ++d;
        return d;
    }

    // Theorem 5 of [35]. Returns true if the branch can be terminated.
    bool early_terminate(const Bits& M, const Bits& C, const Bits& E) {
        if (!E.any()) return false;
        // (i)
        bool hit = false;
        for_each(E, [&](int u) {
            if (hit) return;
            if (!and_any(conf[u], C) && deg_in(u, M) >= k) hit = true;
        });
        if (hit) return true;
        // (ii)
        Bits CE = C;
        for (std::size_t i = 0; i < CE.w.size(); ++i) CE.w[i] |= E.w[i];
        Bits U(n);
        for_each(E, [&](int u) {
            if (!and_any(conf[u], CE)) U.set(u);
        });
        if (!U.any()) return false;
        Bits MU = M;
        for (std::size_t i = 0; i < MU.w.size(); ++i) MU.w[i] |= U.w[i];
        std::vector<int> uv;
        for_each(U, [&](int u) { uv.push_back(u); });
        for (int u : uv) deg_g[u] = deg_in(u, MU);
        stack_.clear();
        for (int u : uv)
            if (deg_g[u] < k) {
                stack_.push_back(u);
                U.reset(u);
                MU.reset(u);
            }
        while (!stack_.empty()) {
            const int v = stack_.back();
            stack_.pop_back();
            for (int u : adj[v])
                if (U.test(u)) {
                    if (--deg_g[u] < k) {
                        U.reset(u);
                        MU.reset(u);
                        stack_.push_back(u);
                    }
                }
        }
        if (!U.any()) return false;
        // keep excluded vertices reachable from M through U
        Bits R(n);
        stack_.clear();
        for_each(M, [&](int v) {
            for (int u : adj[v])
                if (U.test(u) && !R.test(u)) {
                    R.set(u);
                    stack_.push_back(u);
                }
        });
        while (!stack_.empty()) {
            const int v = stack_.back();
            stack_.pop_back();
            for (int u : adj[v])
                if (U.test(u) && !R.test(u)) {
                    R.set(u);
                    stack_.push_back(u);
                }
        }
        return R.any();
    }

    // q-aware (k,k')-core bound on S = M u C (Algorithm 6 of [35]).
    int kk_bound(const Bits& M, const Bits& C) {
        Bits S = M;
        for (std::size_t i = 0; i < S.w.size(); ++i) S.w[i] |= C.w[i];
        std::vector<int> verts;
        for_each(S, [&](int v) { verts.push_back(v); });
        const int s = (int)verts.size();
        int maxd = 0;
        for (int v : verts) {
            alive[v] = 1;
            deg_s[v] = s - 1 - and_count(conf[v], S);
            deg_g[v] = deg_in(v, S);
            if (deg_s[v] > maxd) maxd = deg_s[v];
        }
        std::vector<std::vector<int>> bucket((std::size_t)maxd + 1);
        for (int v : verts) bucket[(std::size_t)deg_s[v]].push_back(v);
        int kp = 0, q_level = -1, cur = 0, remaining = s;
        std::vector<int> cascade;
        while (remaining > 0 && q_level < 0) {
            while (cur <= maxd && bucket[(std::size_t)cur].empty()) ++cur;
            if (cur > maxd) break;
            const int v = bucket[(std::size_t)cur].back();
            bucket[(std::size_t)cur].pop_back();
            if (!alive[v] || deg_s[v] != cur) continue;  // stale entry
            if (deg_s[v] > kp) kp = deg_s[v];
            cascade.clear();
            cascade.push_back(v);
            while (!cascade.empty()) {
                const int x = cascade.back();
                cascade.pop_back();
                if (!alive[x]) continue;
                alive[x] = 0;
                --remaining;
                if (x == q) {
                    q_level = kp;
                    break;
                }
                for (int y : verts) {
                    if (!alive[y] || y == x || conf[x].test(y)) continue;
                    if (deg_s[y] > kp) {
                        --deg_s[y];
                        bucket[(std::size_t)deg_s[y]].push_back(y);
                        if (deg_s[y] < cur) cur = deg_s[y];
                    }
                }
                for (int y : adj[x])
                    if (S.test(y) && alive[y]) {
                        if (--deg_g[y] < k) cascade.push_back(y);
                    }
            }
        }
        for (int v : verts) alive[v] = 0;
        if (q_level < 0) q_level = kp;  // should not happen: q is always peeled eventually
        return q_level + 1;
    }

    void record(const Bits& M, const Bits& C) {
        Bits S = M;
        for (std::size_t i = 0; i < S.w.size(); ++i) S.w[i] |= C.w[i];
        const int sz = S.count();
        if (sz > best_size) {
            best_size = sz;
            best.clear();
            for_each(S, [&](int v) { best.push_back(v); });
            improved = true;
        }
    }

    // Scores of [35]: lambda*Delta1 - Delta2 for the expand and shrink branches
    // of candidate u (two-hop estimates of the vertices removed).
    void score(int u, const Bits& M, const Bits& C, double dpC, double eS, double* s_exp,
               double* s_shr) {
        Bits S = M;
        for (std::size_t i = 0; i < S.w.size(); ++i) S.w[i] |= C.w[i];
        auto eval = [&](Bits& X) -> double {
            // vertices removed one cascade level beyond X
            std::vector<int> xs;
            for_each(X, [&](int v) { xs.push_back(v); });
            std::vector<int> touched;
            for (int x : xs)
                for (int w : adj[x])
                    if (S.test(w) && !X.test(w)) {
                        if (deg_s[w] == 0) touched.push_back(w);  // deg_s reused as a counter
                        ++deg_s[w];
                    }
            for (int w : touched) {
                if (deg_g[w] - deg_s[w] < k) X.set(w);
                deg_s[w] = 0;
            }
            // Delta DP restricted to C
            Bits XC = X;
            for (std::size_t i = 0; i < XC.w.size(); ++i) XC.w[i] &= C.w[i];
            double rem_dp = 0.0, within = 0.0;
            for_each(XC, [&](int x) {
                rem_dp += and_count(conf[x], C);
                within += and_count(conf[x], XC);
            });
            const double ddp = rem_dp - within / 2.0;
            // Delta E on G[M u C]
            double rem_e = 0.0, within_e = 0.0;
            for_each(X, [&](int x) {
                rem_e += deg_g[x];
                for (int w : adj[x])
                    if (X.test(w)) within_e += 1.0;
            });
            const double de = rem_e - within_e / 2.0;
            const double d1 = dpC > 0 ? ddp / dpC : 0.0;
            const double d2 = eS > 0 ? de / eS : 0.0;
            return lambda * d1 - d2;
        };
        Bits Xe(n);
        for (std::size_t i = 0; i < Xe.w.size(); ++i) Xe.w[i] = conf[u].w[i] & C.w[i];
        *s_exp = eval(Xe);
        Bits Xs(n);
        Xs.set(u);
        *s_shr = eval(Xs);
    }

    void search(Bits M, Bits C, Bits E) {
        ++nodes;
        if (check_time()) return;
        if (!peel(M, C)) return;
        // E only keeps vertices similar to all of M (maintained by callers).
        // Direct moves (Remark 1).
        bool moved = true;
        while (moved) {
            moved = false;
            std::vector<int> mv;
            for_each(C, [&](int u) {
                if (!and_any(conf[u], C) && deg_in(u, M) >= k) mv.push_back(u);
            });
            for (int u : mv) {
                if (!C.test(u)) continue;
                C.reset(u);
                M.set(u);
                for (std::size_t i = 0; i < E.w.size(); ++i) E.w[i] &= ~conf[u].w[i];
                moved = true;
            }
        }
        if (early_terminate(M, C, E)) return;
        const int ub = kk_bound(M, C);
        if (ub_root < 0) ub_root = ub;
        if (ub <= best_size) return;
        // DP(C) and candidates not in SF(C)
        double dpC = 0.0;
        std::vector<int> nsf;
        for_each(C, [&](int u) {
            const int c = and_count(conf[u], C);
            dpC += c;
            if (c > 0) nsf.push_back(u);
        });
        dpC /= 2.0;
        if (nsf.empty()) {
            record(M, C);
            return;
        }
        int u = -1;
        bool expand_first = true;
        if (order_mode == 1) {
            int bestc = -1;
            for (int v : nsf) {
                const int c = and_count(conf[v], C);
                if (c > bestc) {
                    bestc = c;
                    u = v;
                }
            }
        } else {
            Bits S = M;
            for (std::size_t i = 0; i < S.w.size(); ++i) S.w[i] |= C.w[i];
            double eS = 0.0;
            for_each(S, [&](int v) {
                deg_g[v] = deg_in(v, S);
                eS += deg_g[v];
                deg_s[v] = 0;
            });
            eS /= 2.0;
            double best_score = -1e300;
            for (int v : nsf) {
                double se, ss;
                score(v, M, C, dpC, eS, &se, &ss);
                const double sc = se > ss ? se : ss;
                if (sc > best_score) {
                    best_score = sc;
                    u = v;
                    expand_first = se >= ss;
                }
            }
        }
        auto expand = [&]() {
            Bits M2 = M, C2 = C, E2 = E;
            M2.set(u);
            C2.reset(u);
            for (std::size_t i = 0; i < C2.w.size(); ++i) {
                C2.w[i] &= ~conf[u].w[i];
                E2.w[i] &= ~conf[u].w[i];
            }
            search(M2, C2, E2);
        };
        auto shrink = [&]() {
            Bits C2 = C, E2 = E;
            C2.reset(u);
            E2.set(u);
            search(M, C2, E2);
        };
        if (expand_first) {
            expand();
            if (timed_out) return;
            shrink();
        } else {
            shrink();
            if (timed_out) return;
            expand();
        }
    }
};

}  // namespace

extern "C" {

int p0_krmax_version() { return 2; }

// Status codes written to out_stats[2]:
//   0 optimal and improved (found best > lb, proven optimal)
//   1 optimal, no solution larger than lb exists
//   2 timeout, improved (best > lb found, not proven)
//   3 timeout, no improvement found
// out_stats: [0] nodes, [1] seconds, [2] status, [3] root upper bound (-1 if not
// reached), [4] size of the candidate set after preprocessing.
int p0_krmax_solve(int n, const int64_t* ptr, const int32_t* idx, int m, const double* X,
                   double tau, int k, int q_local, int lb, double time_limit_s, int order_mode,
                   double rel_eps, int32_t* out_members, int* out_size, double* out_stats) {
    if (n <= 0 || q_local < 0 || q_local >= n || k < 1) return -1;
    Solver sv;
    sv.n = n;
    sv.k = k;
    sv.q = q_local;
    sv.order_mode = order_mode;
    sv.best_size = lb > 0 ? lb : 0;
    sv.time_limit = time_limit_s;
    sv.conf.assign((std::size_t)n, Bits(n));
    sv.adj.assign((std::size_t)n, {});
    sv.deg_g.assign((std::size_t)n, 0);
    sv.deg_s.assign((std::size_t)n, 0);
    sv.alive.assign((std::size_t)n, 0);
    const double t2 = (tau * (1.0 + rel_eps)) * (tau * (1.0 + rel_eps));
    for (int u = 0; u < n; ++u)
        for (int v = u + 1; v < n; ++v)
            if (p0::sqd(X + (int64_t)u * m, X + (int64_t)v * m, m) > t2) {
                sv.conf[(std::size_t)u].set(v);
                sv.conf[(std::size_t)v].set(u);
            }
    for (int v = 0; v < n; ++v)
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p) {
            const int u = idx[p];
            if (u != v && !sv.conf[(std::size_t)v].test(u)) sv.adj[(std::size_t)v].push_back(u);
        }
    Bits M(n), C(n), E(n);
    M.set(q_local);
    for (int v = 0; v < n; ++v)
        if (v != q_local && !sv.conf[(std::size_t)q_local].test(v)) C.set(v);
    // size after preprocessing (k-core of the similar-edge graph, q's component)
    {
        Bits C0 = C;
        if (sv.peel(M, C0))
            out_stats[4] = 1 + C0.count();
        else
            out_stats[4] = 0;
    }
    sv.search(M, C, E);
    out_stats[0] = (double)sv.nodes;
    out_stats[1] = sv.timer.seconds();
    out_stats[3] = sv.ub_root;
    int status;
    if (sv.timed_out)
        status = sv.improved ? 2 : 3;
    else
        status = sv.improved ? 0 : 1;
    out_stats[2] = status;
    if (sv.improved) {
        std::sort(sv.best.begin(), sv.best.end());
        for (std::size_t i = 0; i < sv.best.size(); ++i) out_members[i] = sv.best[i];
        *out_size = (int)sv.best.size();
    } else {
        *out_size = 0;
    }
    return 0;
}

}  // extern "C"
