// LACS P0 kit -- opinion-dynamics stress-test kernels with event tracking.
//
// Models (positions in R^m, Euclidean distance):
//   HK  (model 0): eq. (1) of the paper. Member v averages itself and the
//                  neighbours it accepts, A_v(t) = {u in N(v): d <= theta_acc_v}.
//                  Weights: raw affinities (Wself[v] for v itself, Wedge[p] for
//                  the CSR entry p = (v, u)), normalised over A_v(t) U {v};
//                  NULL means equal weights.
//   SJT (model 1): eq. (2) (Jager-Amblard). The accepted part equals the HK move
//                  above; every rejected neighbour (d > theta_rej_v) pushes v away
//                  with coefficient beta / deg(v).
//   DW  (model 2): asynchronous Deffuant-Weisbuch gossip on the interaction
//                  graph: a uniformly random edge {u,v} interacts; each endpoint
//                  that accepts the other moves a fraction mu towards it.
//   FOLLOW (model 2 of p0_sim_sync): synchronous "follow-one" averaging: each
//                  member moves a fraction gamma towards ONE accepted neighbour
//                  drawn uniformly at each step. A convex but extremely
//                  concentrated, time-varying weighting (close to the adversarial
//                  weights of the characterisation).
// Optional open group (HK/SJT/FOLLOW): member v also has outside neighbours
// at fixed positions oX[o], o in [optr[v], optr[v+1]); those it accepts
// (d <= theta_acc_v) take part in its update exactly like accepted members
// (raw weight Wout[o], or 1), i.e. bounded-confidence dynamics on the whole
// network with the outsiders held still.
// Optional external pull (HK/SJT/FOLLOW): after the update,
//   x_v <- (1 - lambda) x_v + lambda * ext_mean_v   for members with outside
// neighbours (ext_mask[v] = 1), where ext_mean_v is fixed (stubborn outsiders,
// whatever their distance).
//
// Events are tracked on directed interacting pairs (v,u): v accepts u at time t
// iff d(x_v(t), x_u(t)) <= theta_acc_v * (1 + rel_eps). A pair is "broken" if it
// was accepted at t = 0 and is refused at some t >= 1. A pair is "newly
// rejected" if d > theta_rej_v * (1 + rel_eps) at some t >= 1 but not at t = 0.
// The (1 + rel_eps) factor only absorbs floating-point noise; it makes
// acceptance slightly easier and rejection slightly harder.

#include "p0_common.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

enum Metric {
    M_STEPS = 0,
    M_CONVERGED,
    M_NPAIRS,
    M_T0_REFUSED,
    M_BROKEN,
    M_FIRST_BREAK,
    M_T0_REJECTED,
    M_NEW_REJECTED,
    M_FIRST_NEW_REJ,
    M_DIAM0,
    M_DIAM_FINAL,
    M_DIAM_MAX,
    M_N_CLUSTERS,
    M_LARGEST_FRAC,
    M_T_CONSENSUS,
    M_EVER_REJECTED,
    M_OPEN_ACC0,   // members with at least one accepted outside neighbour at t = 0
    M_NUM
};

void finish_common(int n, int m, const double* X, double diam0, double cluster_rel, double* out) {
    const double dfin = std::sqrt(p0::diameter_sq(n, m, X));
    out[M_DIAM_FINAL] = dfin;
    if (out[M_DIAM_FINAL] > out[M_DIAM_MAX]) out[M_DIAM_MAX] = dfin;
    int largest = n;
    int nc = 1;
    if (diam0 > 0.0) {
        const double eps = cluster_rel * diam0;
        nc = p0::clusters(n, m, X, eps * eps, &largest);
    }
    out[M_N_CLUSTERS] = nc;
    out[M_LARGEST_FRAC] = (double)largest / (double)n;
}

}  // namespace

extern "C" {

int p0_dyn_version() { return 5; }
int p0_num_metrics() { return M_NUM; }

// Synchronous HK / SJT / FOLLOW.
// ptr/idx: directed CSR of the interaction graph on the n members (both
// directions present, no self loops). Wself [n] / Wedge [nnz]: raw affinities
// (NULL: equal). optr [n+1] / oX [*, m] / Wout: outside neighbours of the open
// group (NULL: closed group). Returns 0 on success.
int p0_sim_sync(int n, int m, const double* X0, const double* th_acc, const double* th_rej,
                const int64_t* ptr, const int32_t* idx, const double* Wself, const double* Wedge,
                const int64_t* optr, const double* oX, const double* Wout,
                const double* ext_mean, const uint8_t* ext_mask, double lambda_ext, int model, double beta,
                int T_max, double move_tol_rel, double rel_eps, int diam_every, double cons_rel,
                double cluster_rel, double gamma, uint64_t seed, double* out, double* Xout) {
    if (n <= 0 || m <= 0 || T_max < 0 || model < 0 || model > 2) return -1;
    if ((optr == nullptr) != (oX == nullptr)) return -2;
    p0::Rng rng(seed);
    std::vector<const double*> accs;
    if (diam_every < 1) diam_every = 1;
    const int64_t nnz = ptr[n];
    const bool complete = (nnz == (int64_t)n * (int64_t)(n - 1));
    std::vector<double> X(X0, X0 + (int64_t)n * m), Xn((int64_t)n * m);
    std::vector<uint8_t> acc0(nnz, 0), broken(nnz, 0), rej0(nnz, 0), rejnew(nnz, 0);
    std::vector<double> thA2(n), thR2(n);
    for (int v = 0; v < n; ++v) {
        const double a = th_acc[v] * (1.0 + rel_eps);
        const double r = th_rej[v] * (1.0 + rel_eps);
        thA2[v] = a * a;
        thR2[v] = r * r;
    }
    for (int i = 0; i < M_NUM; ++i) out[i] = 0.0;
    out[M_NPAIRS] = (double)nnz;
    out[M_FIRST_BREAK] = -1;
    out[M_FIRST_NEW_REJ] = -1;
    out[M_T_CONSENSUS] = -1;

    const double diam0 = std::sqrt(p0::diameter_sq(n, m, X.data()));
    out[M_DIAM0] = diam0;
    out[M_DIAM_MAX] = diam0;
    if (diam0 <= 0.0) out[M_T_CONSENSUS] = 0;
    const double scale = diam0 > 0.0 ? diam0 : 1.0;
    const double move_tol = move_tol_rel * scale;
    const double cons_thr = cons_rel * diam0;

    std::vector<double> acc_sum(m), rej_sum(m);
    int64_t n_broken = 0, n_new_rej = 0, n_t0_ref = 0, n_t0_rej = 0;
    int steps = 0;
    bool converged = false;
    for (int t = 0;; ++t) {
        double max_move2 = 0.0;
        double maxd2_pairs = 0.0;
        for (int v = 0; v < n; ++v) {
            const double* xv = &X[(int64_t)v * m];
            const double wself = Wself ? Wself[v] : 1.0;
            double wsum = wself;
            for (int j = 0; j < m; ++j) {
                acc_sum[j] = wself * xv[j];
                rej_sum[j] = 0.0;
            }
            int nrej = 0;
            accs.clear();
            const int64_t b = ptr[v], e = ptr[v + 1];
            for (int64_t p = b; p < e; ++p) {
                const int u = idx[p];
                const double* xu = &X[(int64_t)u * m];
                const double d2 = p0::sqd(xv, xu, m);
                if (d2 > maxd2_pairs) maxd2_pairs = d2;
                const bool acc = d2 <= thA2[v];
                const bool rej = d2 > thR2[v];
                if (t == 0) {
                    acc0[p] = acc;
                    rej0[p] = rej;
                    if (!acc) ++n_t0_ref;
                    if (rej) ++n_t0_rej;
                } else {
                    if (acc0[p] && !acc && !broken[p]) {
                        broken[p] = 1;
                        ++n_broken;
                        if (out[M_FIRST_BREAK] < 0) out[M_FIRST_BREAK] = t;
                    }
                    if (rej && !rej0[p] && !rejnew[p]) {
                        rejnew[p] = 1;
                        ++n_new_rej;
                        if (out[M_FIRST_NEW_REJ] < 0) out[M_FIRST_NEW_REJ] = t;
                    }
                }
                if (acc) {
                    if (model == 2) {
                        accs.push_back(xu);
                    } else {
                        const double w = Wedge ? Wedge[p] : 1.0;
                        wsum += w;
                        for (int j = 0; j < m; ++j) acc_sum[j] += w * xu[j];
                    }
                }
                if (model == 1 && rej) {
                    for (int j = 0; j < m; ++j) rej_sum[j] += xu[j] - xv[j];
                    ++nrej;
                }
            }
            if (optr) {
                // accepted outside neighbours (fixed positions) join the update
                bool any = false;
                for (int64_t o = optr[v]; o < optr[v + 1]; ++o) {
                    const double* y = oX + o * m;
                    if (p0::sqd(xv, y, m) > thA2[v]) continue;
                    any = true;
                    if (model == 2) {
                        accs.push_back(y);
                    } else {
                        const double w = Wout ? Wout[o] : 1.0;
                        wsum += w;
                        for (int j = 0; j < m; ++j) acc_sum[j] += w * y[j];
                    }
                }
                if (t == 0 && any) out[M_OPEN_ACC0] += 1.0;
            }
            double* xnv = &Xn[(int64_t)v * m];
            if (model == 2) {
                // follow-one: move a fraction gamma towards one accepted neighbour
                if (!accs.empty()) {
                    const double* xu = accs[(std::size_t)rng.below(accs.size())];
                    for (int j = 0; j < m; ++j) xnv[j] = (1.0 - gamma) * xv[j] + gamma * xu[j];
                } else {
                    for (int j = 0; j < m; ++j) xnv[j] = xv[j];
                }
            } else {
                for (int j = 0; j < m; ++j) xnv[j] = acc_sum[j] / wsum;
            }
            const int64_t deg = e - b;
            if (model == 1 && nrej > 0 && deg > 0) {
                const double c = beta / (double)deg;
                for (int j = 0; j < m; ++j) xnv[j] -= c * rej_sum[j];
            }
            if (ext_mean && ext_mask && ext_mask[v] && lambda_ext > 0.0) {
                const double* ex = &ext_mean[(int64_t)v * m];
                for (int j = 0; j < m; ++j) xnv[j] = (1.0 - lambda_ext) * xnv[j] + lambda_ext * ex[j];
            }
            const double mv2 = p0::sqd(xnv, xv, m);
            if (mv2 > max_move2) max_move2 = mv2;
        }
        // Diameter at time t (positions X).
        if (t > 0) {
            double dt = -1.0;
            if (complete) {
                dt = std::sqrt(maxd2_pairs);
            } else if (t % diam_every == 0) {
                dt = std::sqrt(p0::diameter_sq(n, m, X.data()));
            }
            if (dt >= 0.0) {
                if (dt > out[M_DIAM_MAX]) out[M_DIAM_MAX] = dt;
                if (out[M_T_CONSENSUS] < 0 && dt <= cons_thr) out[M_T_CONSENSUS] = t;
            }
        }
        if (t >= T_max) break;
        if (std::sqrt(max_move2) <= move_tol) {
            converged = true;
            break;
        }
        std::swap(X, Xn);
        ++steps;
    }
    out[M_STEPS] = steps;
    out[M_CONVERGED] = converged ? 1.0 : 0.0;
    out[M_T0_REFUSED] = (double)n_t0_ref;
    out[M_BROKEN] = (double)n_broken;
    out[M_T0_REJECTED] = (double)n_t0_rej;
    out[M_NEW_REJECTED] = (double)n_new_rej;
    out[M_EVER_REJECTED] = (double)(n_t0_rej + n_new_rej);
    finish_common(n, m, X.data(), diam0, cluster_rel, out);
    if (out[M_T_CONSENSUS] < 0 && diam0 > 0.0 && out[M_DIAM_FINAL] <= cons_thr)
        out[M_T_CONSENSUS] = steps;
    if (Xout) std::memcpy(Xout, X.data(), sizeof(double) * (std::size_t)n * (std::size_t)m);
    return 0;
}

// Asynchronous DW gossip. rev[p] is the CSR index of the reverse pair of p.
// One round = nnz/2 micro-steps (each undirected edge once on average).
// Time stamps (first_break, t_consensus, steps) are in rounds.
int p0_sim_dw(int n, int m, const double* X0, const double* th_acc, const double* th_rej,
              const int64_t* ptr, const int32_t* idx, const int64_t* rev, double mu, int rounds,
              uint64_t seed, double move_tol_rel, double rel_eps, int diam_every,
              double cons_rel, double cluster_rel, double* out, double* Xout) {
    if (n <= 0 || m <= 0 || rounds < 0) return -1;
    if (diam_every < 1) diam_every = 1;
    const int64_t nnz = ptr[n];
    std::vector<double> X(X0, X0 + (int64_t)n * m), Xstart((int64_t)n * m), xv_old(m);
    std::vector<int32_t> src(nnz);
    for (int v = 0; v < n; ++v)
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p) src[p] = v;
    std::vector<uint8_t> acc0(nnz, 0), broken(nnz, 0), rej0(nnz, 0), rejnew(nnz, 0);
    std::vector<double> thA2(n), thR2(n);
    for (int v = 0; v < n; ++v) {
        const double a = th_acc[v] * (1.0 + rel_eps);
        const double r = th_rej[v] * (1.0 + rel_eps);
        thA2[v] = a * a;
        thR2[v] = r * r;
    }
    for (int i = 0; i < M_NUM; ++i) out[i] = 0.0;
    out[M_NPAIRS] = (double)nnz;
    out[M_FIRST_BREAK] = -1;
    out[M_FIRST_NEW_REJ] = -1;
    out[M_T_CONSENSUS] = -1;
    const double diam0 = std::sqrt(p0::diameter_sq(n, m, X.data()));
    out[M_DIAM0] = diam0;
    out[M_DIAM_MAX] = diam0;
    if (diam0 <= 0.0) out[M_T_CONSENSUS] = 0;
    const double scale = diam0 > 0.0 ? diam0 : 1.0;
    const double move_tol = move_tol_rel * scale;
    const double cons_thr = cons_rel * diam0;

    int64_t n_broken = 0, n_new_rej = 0, n_t0_ref = 0, n_t0_rej = 0;
    for (int v = 0; v < n; ++v)
        for (int64_t p = ptr[v]; p < ptr[v + 1]; ++p) {
            const double d2 = p0::sqd(&X[(int64_t)v * m], &X[(int64_t)idx[p] * m], m);
            acc0[p] = d2 <= thA2[v];
            rej0[p] = d2 > thR2[v];
            if (!acc0[p]) ++n_t0_ref;
            if (rej0[p]) ++n_t0_rej;
        }

    p0::Rng rng(seed);
    const int64_t micro = nnz / 2 > 0 ? nnz / 2 : 0;
    int done_rounds = 0;
    bool converged = (nnz == 0);
    auto check_incident = [&](int a, double t_now) {
        const double* xa = &X[(int64_t)a * m];
        for (int64_t p = ptr[a]; p < ptr[a + 1]; ++p) {
            const int w = idx[p];
            const double d2 = p0::sqd(xa, &X[(int64_t)w * m], m);
            const int64_t q = rev[p];
            // pair a -> w (a's view) and w -> a (w's view)
            if (acc0[p] && !broken[p] && d2 > thA2[a]) {
                broken[p] = 1;
                ++n_broken;
                if (out[M_FIRST_BREAK] < 0) out[M_FIRST_BREAK] = t_now;
            }
            if (acc0[q] && !broken[q] && d2 > thA2[w]) {
                broken[q] = 1;
                ++n_broken;
                if (out[M_FIRST_BREAK] < 0) out[M_FIRST_BREAK] = t_now;
            }
            if (!rej0[p] && !rejnew[p] && d2 > thR2[a]) {
                rejnew[p] = 1;
                ++n_new_rej;
                if (out[M_FIRST_NEW_REJ] < 0) out[M_FIRST_NEW_REJ] = t_now;
            }
            if (!rej0[q] && !rejnew[q] && d2 > thR2[w]) {
                rejnew[q] = 1;
                ++n_new_rej;
                if (out[M_FIRST_NEW_REJ] < 0) out[M_FIRST_NEW_REJ] = t_now;
            }
        }
    };
    for (int r = 0; r < rounds && !converged; ++r) {
        Xstart = X;
        for (int64_t s = 0; s < micro; ++s) {
            const int64_t p = (int64_t)rng.below((uint64_t)nnz);
            const int v = src[p], u = idx[p];
            double* xv = &X[(int64_t)v * m];
            double* xu = &X[(int64_t)u * m];
            const double d2 = p0::sqd(xv, xu, m);
            const bool av = d2 <= thA2[v];
            const bool au = d2 <= thA2[u];
            if (!av && !au) continue;
            for (int j = 0; j < m; ++j) xv_old[j] = xv[j];
            if (av)
                for (int j = 0; j < m; ++j) xv[j] += mu * (xu[j] - xv[j]);
            if (au)
                for (int j = 0; j < m; ++j) xu[j] += mu * (xv_old[j] - xu[j]);
            const double t_now = (double)r + (double)(s + 1) / (double)micro;
            check_incident(v, t_now);
            check_incident(u, t_now);
        }
        ++done_rounds;
        double max_move2 = 0.0;
        for (int v = 0; v < n; ++v) {
            const double d2 = p0::sqd(&X[(int64_t)v * m], &Xstart[(int64_t)v * m], m);
            if (d2 > max_move2) max_move2 = d2;
        }
        if (done_rounds % diam_every == 0) {
            const double dt = std::sqrt(p0::diameter_sq(n, m, X.data()));
            if (dt > out[M_DIAM_MAX]) out[M_DIAM_MAX] = dt;
            if (out[M_T_CONSENSUS] < 0 && dt <= cons_thr) out[M_T_CONSENSUS] = done_rounds;
        }
        if (std::sqrt(max_move2) <= move_tol) converged = true;
    }
    out[M_STEPS] = done_rounds;
    out[M_CONVERGED] = converged ? 1.0 : 0.0;
    out[M_T0_REFUSED] = (double)n_t0_ref;
    out[M_BROKEN] = (double)n_broken;
    out[M_T0_REJECTED] = (double)n_t0_rej;
    out[M_NEW_REJECTED] = (double)n_new_rej;
    out[M_EVER_REJECTED] = (double)(n_t0_rej + n_new_rej);
    finish_common(n, m, X.data(), diam0, cluster_rel, out);
    if (out[M_T_CONSENSUS] < 0 && diam0 > 0.0 && out[M_DIAM_FINAL] <= cons_thr)
        out[M_T_CONSENSUS] = done_rounds;
    if (Xout) std::memcpy(Xout, X.data(), sizeof(double) * (std::size_t)n * (std::size_t)m);
    return 0;
}

}  // extern "C"
