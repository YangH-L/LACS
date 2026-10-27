// LACS core: graph I/O, core decomposition and latitude-aware community search.
//
// Problem (LCC, "latitude-consistent community"): given an undirected graph G,
// positions x_v in R^m, per-vertex tolerances theta_v, a query q and integer k,
// find a maximum vertex set H with
//   (1) q in H, G[H] connected and min-degree >= k (a connected k-core), and
//   (2) max_{u,v in H} ||x_u - x_v|| <= min_{w in H} theta_w.
// With theta = theta_acc this is C-LCC (consensus-guaranteed); with
// theta = theta_rej it is R-LCC (rejection-free). The engine is agnostic: the
// caller passes whichever tolerance vector it wants.
//
// Algorithms (see the paper, Section V):
//   kcore        connected k-core component of q (structure-only baseline)
//   ballcore_f   feasible solution: members within tau/2 of q, tau swept
//   ballcore_r   (1,2)-bicriteria: size >= OPT, diameter <= 2 * min theta
//   greedy       conflict-peeling heuristic, always feasible
//   exact        LatSweep: tau sweep + branch-and-bound, optimal unless timeout
//   kr_exact / kr_greedy           global distance threshold r ((k,r)-core)
//   pairwise_exact / pairwise_greedy  d(u,v) <= min(theta_u, theta_v) only
//   edgeacc      k-core after dropping edges that are not mutually accepted
//   sac_inc      spatial-aware community search baseline (Fang et al., PVLDB 2017,
//                AppInc): connected k-core containing q in the smallest ball centred
//                at q that holds one; positions only, no tolerances (tau = radius).
//                With min_size > 0: the smallest ball whose k-core with q has at least
//                min_size members (a size-matched spatial baseline)
//
// Candidate generation (v0.3). Two sound reductions, both on by default:
//   * compatibility index: G_theta keeps the edges with d(u,v) <= min(theta_u, theta_v).
//     Every edge inside a feasible community is compatible (d(u,v) <= diameter <=
//     min theta), so a feasible community is a connected k-core of G_theta: it lies
//     in the k-core of G_theta (core numbers computed once per process).
//   * query-local candidates: instead of materialising q's whole k-core component,
//     grow the region from q through vertices v with d(q,v) <= min(theta_v, theta_q)
//     (the interval lemma), along compatible edges for the consensus-type algorithms.
//     A feasible community is connected and inside this ball, so it lies in the region.
// --no-index / --no-local switch them off (ablations; results must not change).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <tuple>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lacs {

// ---------------------------------------------------------------- I/O ----

struct Graph {
  int64_t n = 0;
  std::vector<int64_t> off;  // size n+1
  std::vector<int32_t> adj;  // size 2m, symmetric, no self loops, no duplicates
  int64_t deg(int64_t v) const { return off[v + 1] - off[v]; }
  int64_t num_edges() const { return static_cast<int64_t>(adj.size()) / 2; }
};

struct Matrix {
  int64_t rows = 0, cols = 0;
  std::vector<float> data;
  const float* row(int64_t i) const { return data.data() + i * cols; }
};

static const char kGraphMagic[8] = {'L', 'A', 'C', 'S', 'G', 'R', '0', '1'};
static const char kMatrixMagic[8] = {'L', 'A', 'C', 'S', 'M', 'T', '0', '1'};

inline void read_exact(std::ifstream& in, void* dst, size_t bytes, const std::string& path) {
  in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(bytes));
  if (!in) throw std::runtime_error("truncated file: " + path);
}

inline Graph read_graph(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open graph: " + path);
  char magic[8];
  read_exact(in, magic, 8, path);
  if (std::memcmp(magic, kGraphMagic, 8) != 0) throw std::runtime_error("bad graph magic: " + path);
  Graph g;
  int64_t nnz = 0;
  read_exact(in, &g.n, sizeof(int64_t), path);
  read_exact(in, &nnz, sizeof(int64_t), path);
  g.off.resize(static_cast<size_t>(g.n + 1));
  g.adj.resize(static_cast<size_t>(nnz));
  read_exact(in, g.off.data(), sizeof(int64_t) * g.off.size(), path);
  if (nnz > 0) read_exact(in, g.adj.data(), sizeof(int32_t) * g.adj.size(), path);
  if (g.off.back() != nnz) throw std::runtime_error("graph offsets inconsistent: " + path);
  return g;
}

inline void write_graph(const std::string& path, const Graph& g) {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write graph: " + path);
  int64_t nnz = static_cast<int64_t>(g.adj.size());
  out.write(kGraphMagic, 8);
  out.write(reinterpret_cast<const char*>(&g.n), sizeof(int64_t));
  out.write(reinterpret_cast<const char*>(&nnz), sizeof(int64_t));
  out.write(reinterpret_cast<const char*>(g.off.data()), sizeof(int64_t) * g.off.size());
  if (nnz > 0) out.write(reinterpret_cast<const char*>(g.adj.data()), sizeof(int32_t) * g.adj.size());
}

inline Matrix read_matrix(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open matrix: " + path);
  char magic[8];
  read_exact(in, magic, 8, path);
  if (std::memcmp(magic, kMatrixMagic, 8) != 0) throw std::runtime_error("bad matrix magic: " + path);
  Matrix m;
  read_exact(in, &m.rows, sizeof(int64_t), path);
  read_exact(in, &m.cols, sizeof(int64_t), path);
  m.data.resize(static_cast<size_t>(m.rows * m.cols));
  if (!m.data.empty()) read_exact(in, m.data.data(), sizeof(float) * m.data.size(), path);
  return m;
}

// Build CSR from an undirected edge list (u,v), dropping self loops and duplicates.
inline Graph build_csr(int64_t n, std::vector<std::pair<int32_t, int32_t>>& edges) {
  std::vector<uint64_t> keys;
  keys.reserve(edges.size() * 2);
  for (auto& e : edges) {
    if (e.first == e.second) continue;
    if (e.first < 0 || e.second < 0 || e.first >= n || e.second >= n)
      throw std::runtime_error("edge endpoint out of range");
    keys.push_back((static_cast<uint64_t>(e.first) << 32) | static_cast<uint32_t>(e.second));
    keys.push_back((static_cast<uint64_t>(e.second) << 32) | static_cast<uint32_t>(e.first));
  }
  std::vector<std::pair<int32_t, int32_t>>().swap(edges);
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  Graph g;
  g.n = n;
  g.off.assign(static_cast<size_t>(n + 1), 0);
  g.adj.resize(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    int64_t u = static_cast<int64_t>(keys[i] >> 32);
    g.off[u + 1]++;
    g.adj[i] = static_cast<int32_t>(keys[i] & 0xffffffffULL);
  }
  for (int64_t v = 0; v < n; ++v) g.off[v + 1] += g.off[v];
  return g;
}

// ------------------------------------------------------- core numbers ----

// Batagelj-Zaversnik O(m) core decomposition.
inline std::vector<int32_t> core_numbers(const Graph& g) {
  const int64_t n = g.n;
  std::vector<int32_t> deg(static_cast<size_t>(n));
  int32_t md = 0;
  for (int64_t v = 0; v < n; ++v) {
    deg[v] = static_cast<int32_t>(g.deg(v));
    md = std::max(md, deg[v]);
  }
  std::vector<int64_t> bin(static_cast<size_t>(md) + 1, 0);
  for (int64_t v = 0; v < n; ++v) bin[deg[v]]++;
  int64_t start = 0;
  for (int32_t d = 0; d <= md; ++d) {
    int64_t num = bin[d];
    bin[d] = start;
    start += num;
  }
  std::vector<int64_t> pos(static_cast<size_t>(n)), vert(static_cast<size_t>(n));
  for (int64_t v = 0; v < n; ++v) {
    pos[v] = bin[deg[v]];
    vert[pos[v]] = v;
    bin[deg[v]]++;
  }
  for (int32_t d = md; d >= 1; --d) bin[d] = bin[d - 1];
  if (md >= 0) bin[0] = 0;
  for (int64_t i = 0; i < n; ++i) {
    int64_t v = vert[i];
    for (int64_t e = g.off[v]; e < g.off[v + 1]; ++e) {
      int64_t u = g.adj[e];
      if (deg[u] > deg[v]) {
        int32_t du = deg[u];
        int64_t pu = pos[u];
        int64_t pw = bin[du];
        int64_t w = vert[pw];
        if (u != w) {
          pos[u] = pw;
          vert[pu] = w;
          pos[w] = pu;
          vert[pw] = u;
        }
        bin[du]++;
        deg[u]--;
      }
    }
  }
  return deg;
}

// ------------------------------------------------------------ bitsets ----

struct Bitset {
  std::vector<uint64_t> w;
  int64_t nbits = 0;
  Bitset() = default;
  explicit Bitset(int64_t n) : w(static_cast<size_t>((n + 63) / 64), 0ULL), nbits(n) {}
  void set(int64_t i) { w[i >> 6] |= (1ULL << (i & 63)); }
  void reset(int64_t i) { w[i >> 6] &= ~(1ULL << (i & 63)); }
  bool test(int64_t i) const { return (w[i >> 6] >> (i & 63)) & 1ULL; }
  int64_t count() const {
    int64_t c = 0;
    for (auto x : w) c += __builtin_popcountll(x);
    return c;
  }
  bool any() const {
    for (auto x : w)
      if (x) return true;
    return false;
  }
  template <class F>
  void for_each(F f) const {
    for (size_t b = 0; b < w.size(); ++b) {
      uint64_t x = w[b];
      while (x) {
        int t = __builtin_ctzll(x);
        f(static_cast<int64_t>(b * 64 + t));
        x &= x - 1;
      }
    }
  }
};

inline bool intersects(const Bitset& a, const Bitset& b) {
  for (size_t i = 0; i < a.w.size(); ++i)
    if (a.w[i] & b.w[i]) return true;
  return false;
}
inline int64_t and_count(const Bitset& a, const Bitset& b) {
  int64_t c = 0;
  for (size_t i = 0; i < a.w.size(); ++i) c += __builtin_popcountll(a.w[i] & b.w[i]);
  return c;
}

// ------------------------------------------------------------ results ----

struct Options {
  int k = 3;
  double time_limit_s = 600.0;     // per query, exact algorithms
  int64_t max_exact_size = 3000;   // candidate sets larger than this are not searched (bounded only)
  int64_t max_greedy_size = 20000; // greedy skips candidate sets larger than this
  int greedy_max_tau = 20;         // number of tau values greedy evaluates (by upper bound)
  bool use_ub_order = true;        // process tau by decreasing upper bound, stop early
  bool use_color_bound = true;     // greedy-coloring bound inside branch-and-bound
  bool use_seed = true;            // seed exact search with ballcore_f + greedy
  bool use_compat_bound = true;    // compatibility-degree pruning against the incumbent
  bool use_edge_filter = true;     // drop graph edges between conflicting vertices before peeling
  bool use_local = true;           // query-local candidate region instead of q's whole k-core component
  bool use_index = true;           // compatibility-core index (k-core of G_theta)
  double r = -1.0;                 // global threshold for kr_* algorithms
  int64_t min_size = 0;            // sac_inc: smallest ball whose k-core with q has at least this many members
  int64_t diam_cap = 5000;         // skip exact diameter check for larger results
  bool color_order_degree = true;  // v0.5: colour by increasing conflict count within P (Tomita-Seki); false = index order
  bool tau_order_bound = true;     // v0.5: exact refines thresholds best-first by upper bound; false = v0.4 count order
  bool compute_ub = true;          // v0.5: exact algorithms report an upper bound on OPT when optimality is not proven
  double ub_time_s = -1.0;         // time for that bound after the search; < 0: max(5 s, 10% of time_limit_s)
  int64_t max_bound_size = 20000;  // largest candidate set that gets a colouring bound in the bound pass
};

struct Result {
  int64_t q = -1;
  std::string algo;
  int k = 0;
  std::vector<int64_t> members;
  std::string status = "ok";
  double time_ms = 0.0;
  double tau = std::numeric_limits<double>::quiet_NaN();
  double diam = std::numeric_limits<double>::quiet_NaN();
  double min_theta = std::numeric_limits<double>::quiet_NaN();
  bool feasible = false;           // diam <= min theta (+eps)
  int64_t s0_size = 0;             // connected k-core component of q
  int64_t n_candidates = 0;        // vertices that pass the query filter
  int64_t n_tau = 0, n_tau_eval = 0;
  int64_t bb_nodes = 0;
  double index_ms = 0.0;           // one-off index construction triggered by this query (not in time_ms)
  int64_t index_edges = -1;        // edges of the compatibility graph G_theta (-1: index not built)
  // v0.5. Upper bound on OPT: for exact algorithms the size itself when optimal, otherwise from the
  // bound pass (status timeout / too_large); for ballcore_r |B| >= OPT when its sweep finished
  // (Theorem 4); -1 when no bound is available.
  int64_t ub = -1;
  double ub_ms = 0.0;              // time of the bound pass, after the search (not included in time_ms)
  int64_t n_tau_open = 0;          // thresholds whose bound exceeded the incumbent when the bound pass started
};

inline std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o;
}

inline std::string fmt_double(double x) {
  if (std::isnan(x)) return "null";
  if (std::isinf(x)) return x > 0 ? "1e308" : "-1e308";
  std::ostringstream ss;
  ss.precision(10);
  ss << x;
  return ss.str();
}

inline std::string to_json(const Result& r, bool with_members) {
  std::ostringstream o;
  o << "{\"q\":" << r.q << ",\"algo\":\"" << json_escape(r.algo) << "\",\"k\":" << r.k
    << ",\"size\":" << r.members.size() << ",\"status\":\"" << json_escape(r.status) << "\""
    << ",\"time_ms\":" << fmt_double(r.time_ms) << ",\"tau\":" << fmt_double(r.tau)
    << ",\"diam\":" << fmt_double(r.diam) << ",\"min_theta\":" << fmt_double(r.min_theta)
    << ",\"feasible\":" << (r.feasible ? "true" : "false") << ",\"s0_size\":" << r.s0_size
    << ",\"n_candidates\":" << r.n_candidates << ",\"n_tau\":" << r.n_tau
    << ",\"n_tau_eval\":" << r.n_tau_eval << ",\"bb_nodes\":" << r.bb_nodes
    << ",\"index_ms\":" << fmt_double(r.index_ms) << ",\"index_edges\":" << r.index_edges
    << ",\"ub\":" << r.ub << ",\"ub_ms\":" << fmt_double(r.ub_ms) << ",\"n_tau_open\":" << r.n_tau_open;
  if (with_members) {
    o << ",\"members\":[";
    for (size_t i = 0; i < r.members.size(); ++i) o << (i ? "," : "") << r.members[i];
    o << "]";
  }
  o << "}";
  return o.str();
}

// ----------------------------------------------------------- searcher ----

class Searcher {
 public:
  Searcher(const Graph& g, const Matrix& pos, const std::vector<float>& theta)
      : g_(g), pos_(pos), theta_(theta) {
    if (pos_.rows != g_.n) throw std::runtime_error("positions rows != graph n");
    if (static_cast<int64_t>(theta_.size()) != g_.n) throw std::runtime_error("theta size != graph n");
    for (auto& t : theta_)
      if (!std::isfinite(t) || t < 0) t = 0.0f;
    core_ = core_numbers(g_);
    loc_.assign(static_cast<size_t>(g_.n), -1);
    mark_.assign(static_cast<size_t>(g_.n), 0);
  }

  // Build the compatibility index now (otherwise built on first use). Returns its build time in ms.
  double build_index() {
    if (index_ready_) return 0.0;
    auto t0 = Clock::now();
    const int64_t n = g_.n;
    std::vector<uint8_t> ok(g_.adj.size(), 0);
    for (int64_t v = 0; v < n; ++v) {
      for (int64_t e = g_.off[v]; e < g_.off[v + 1]; ++e) {
        const int64_t u = g_.adj[e];
        if (u <= v) continue;
        if (dist(v, u) > std::min(static_cast<double>(theta_[v]), static_cast<double>(theta_[u]))) continue;
        ok[e] = 1;
        auto b = g_.adj.begin() + g_.off[u], en = g_.adj.begin() + g_.off[u + 1];
        auto it = std::lower_bound(b, en, static_cast<int32_t>(v));
        if (it != en && *it == v) {
          ok[static_cast<size_t>(it - g_.adj.begin())] = 1;
        } else {  // adjacency lists not sorted: linear scan
          for (int64_t e2 = g_.off[u]; e2 < g_.off[u + 1]; ++e2)
            if (g_.adj[e2] == v) {
              ok[e2] = 1;
              break;
            }
        }
      }
    }
    hc_.n = n;
    hc_.off.assign(static_cast<size_t>(n + 1), 0);
    for (int64_t v = 0; v < n; ++v) {
      int64_t c = 0;
      for (int64_t e = g_.off[v]; e < g_.off[v + 1]; ++e) c += ok[e];
      hc_.off[v + 1] = hc_.off[v] + c;
    }
    hc_.adj.resize(static_cast<size_t>(hc_.off[n]));
    for (int64_t v = 0; v < n; ++v) {
      int64_t p = hc_.off[v];
      for (int64_t e = g_.off[v]; e < g_.off[v + 1]; ++e)
        if (ok[e]) hc_.adj[p++] = g_.adj[e];
    }
    ccore_ = core_numbers(hc_);
    index_ready_ = true;
    index_ms_ = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return index_ms_;
  }
  double index_ms() const { return index_ms_; }
  int64_t index_edges() const { return hc_.num_edges(); }

  int max_core() const {
    int m = 0;
    for (auto c : core_) m = std::max(m, c);
    return m;
  }
  const std::vector<int32_t>& core() const { return core_; }

  double dist(int64_t a, int64_t b) const {
    const float* x = pos_.row(a);
    const float* y = pos_.row(b);
    double s = 0.0;
    for (int64_t j = 0; j < pos_.cols; ++j) {
      double d = static_cast<double>(x[j]) - static_cast<double>(y[j]);
      s += d * d;
    }
    return std::sqrt(s);
  }

  Result run(const std::string& algo, int64_t q, const Options& opt) {
    Result r;
    r.q = q;
    r.algo = algo;
    r.k = opt.k;
    const bool uses_index = algo == "ballcore_f" || algo == "greedy" || algo == "exact" ||
                            algo == "pairwise_exact" || algo == "pairwise_greedy" || algo == "edgeacc";
    // one-off preprocessing, reported separately from the per-query search time
    if (opt.use_index && uses_index && !index_ready_ && q >= 0 && q < g_.n) r.index_ms = build_index();
    if (index_ready_) r.index_edges = index_edges();
    auto t0 = Clock::now();
    if (q < 0 || q >= g_.n) {
      r.status = "bad_query";
    } else if (core_[q] < opt.k) {
      r.status = "not_in_kcore";
      if (algo == "exact" || algo == "ballcore_r" || algo == "kr_exact" || algo == "pairwise_exact") r.ub = 0;
    } else {
      deadline_ = t0 + std::chrono::duration_cast<Clock::duration>(
                           std::chrono::duration<double>(opt.time_limit_s));
      timed_out_ = false;
      greedy_cap_ = opt.max_greedy_size;
      greedy_skipped_ = false;
      if (algo == "kcore") {
        r.members = kcore_component(q, opt.k);
        r.s0_size = static_cast<int64_t>(r.members.size());
      } else if (algo == "ballcore_f" || algo == "ballcore_r" || algo == "greedy" || algo == "exact") {
        run_latitude(algo, q, opt, r);
      } else if (algo == "kr_exact" || algo == "kr_greedy" || algo == "pairwise_exact" ||
                 algo == "pairwise_greedy") {
        run_threshold(algo, q, opt, r);
      } else if (algo == "edgeacc") {
        run_edgeacc(q, opt, r);
      } else if (algo == "sac_inc") {
        run_sac(q, opt, r);
      } else {
        throw std::runtime_error("unknown algo: " + algo);
      }
      if (greedy_skipped_ && (r.status == "ok" || r.status == "optimal")) r.status = "greedy_skipped";
    }
    // timing covers the search only, not the bound pass (r.ub_ms) or the post-hoc diameter check below
    r.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() - r.ub_ms;
    std::sort(r.members.begin(), r.members.end());
    finalize(r, opt.diam_cap);
    return r;
  }

 private:
  using Clock = std::chrono::steady_clock;
  const Graph& g_;
  const Matrix& pos_;
  std::vector<float> theta_;
  std::vector<int32_t> core_;
  std::vector<int32_t> loc_;  // global -> local index, -1 if absent (reset after use)
  std::vector<int8_t> mark_;  // region growing: 0 unseen, 1 in region, 2 rejected, 3 vertex ok but not reached
  Graph hc_;                  // compatibility graph G_theta (built lazily)
  std::vector<int32_t> ccore_;
  bool index_ready_ = false;
  double index_ms_ = 0.0;
  Clock::time_point deadline_;
  bool timed_out_ = false;
  int64_t greedy_cap_ = 20000;
  bool greedy_skipped_ = false;

  // Local induced subgraph over a candidate list (global ids), with local CSR.
  struct Local {
    std::vector<int64_t> ids;   // local -> global
    std::vector<int64_t> off;   // local CSR
    std::vector<int32_t> adj;
    int64_t size() const { return static_cast<int64_t>(ids.size()); }
  };

  Local make_local(const std::vector<int64_t>& ids) { return make_local(ids, g_); }

  // Induced subgraph of h over ids.
  Local make_local(const std::vector<int64_t>& ids, const Graph& h) {
    Local L;
    L.ids = ids;
    for (size_t i = 0; i < ids.size(); ++i) loc_[ids[i]] = static_cast<int32_t>(i);
    L.off.assign(ids.size() + 1, 0);
    for (size_t i = 0; i < ids.size(); ++i) {
      int64_t v = ids[i];
      int64_t c = 0;
      for (int64_t e = h.off[v]; e < h.off[v + 1]; ++e)
        if (loc_[h.adj[e]] >= 0) ++c;
      L.off[i + 1] = L.off[i] + c;
    }
    L.adj.resize(static_cast<size_t>(L.off.back()));
    for (size_t i = 0; i < ids.size(); ++i) {
      int64_t v = ids[i];
      int64_t p = L.off[i];
      for (int64_t e = h.off[v]; e < h.off[v + 1]; ++e) {
        int32_t lu = loc_[h.adj[e]];
        if (lu >= 0) L.adj[p++] = lu;
      }
    }
    for (size_t i = 0; i < ids.size(); ++i) loc_[ids[i]] = -1;
    return L;
  }

  // Connected k-core component of q in the whole graph.
  std::vector<int64_t> kcore_component(int64_t q, int k) {
    std::vector<int64_t> out;
    std::vector<int64_t> stack{q};
    loc_[q] = 0;  // loc_ doubles as the visited marker here; reset below
    while (!stack.empty()) {
      int64_t v = stack.back();
      stack.pop_back();
      out.push_back(v);
      for (int64_t e = g_.off[v]; e < g_.off[v + 1]; ++e) {
        int64_t u = g_.adj[e];
        if (core_[u] >= k && loc_[u] < 0) {
          loc_[u] = 0;
          stack.push_back(u);
        }
      }
    }
    for (auto v : out) loc_[v] = -1;
    return out;
  }

  // Connected component of q among vertices with corev >= k in graph h.
  std::vector<int64_t> component_on(const Graph& h, const std::vector<int32_t>& corev, int64_t q, int k) {
    if (corev[q] < k) return {};
    return grow_region(q, h, [&](int64_t u) { return corev[u] >= k; }, [](int64_t, int64_t) { return true; });
  }

  // Region grown from q along edges (v,u) of h with eok(v,u), through vertices with vok(u).
  template <class VOk, class EOk>
  std::vector<int64_t> grow_region(int64_t q, const Graph& h, VOk vok, EOk eok) {
    std::vector<int64_t> out{q}, touched{q};
    mark_[q] = 1;
    for (size_t i = 0; i < out.size(); ++i) {
      const int64_t v = out[i];
      for (int64_t e = h.off[v]; e < h.off[v + 1]; ++e) {
        const int64_t u = h.adj[e];
        const int8_t m = mark_[u];
        if (m == 1 || m == 2) continue;
        if (m == 0) {
          touched.push_back(u);
          if (!vok(u)) {
            mark_[u] = 2;
            continue;
          }
          mark_[u] = 3;
        }
        if (!eok(v, u)) continue;  // u may still be reached through another edge
        mark_[u] = 1;
        out.push_back(u);
      }
    }
    for (auto v : touched) mark_[v] = 0;
    return out;
  }

  bool compatible(int64_t a, int64_t b) const {
    return dist(a, b) <= std::min(static_cast<double>(theta_[a]), static_cast<double>(theta_[b]));
  }

  // Candidates for the consensus-type algorithms (exact, greedy, ballcore_f, pairwise_*):
  // every feasible community containing q is inside the returned set (see header).
  std::vector<int64_t> candidates_c(int64_t q, int k, double scale, const Options& opt) {
    const double tq = theta_[q];
    auto in_ball = [&](int64_t u) { return scale * dist(q, u) <= std::min(static_cast<double>(theta_[u]), tq); };
    if (opt.use_index) {
      if (ccore_[q] < k) return {};
      if (!opt.use_local) return component_on(hc_, ccore_, q, k);
      return grow_region(q, hc_, [&](int64_t u) { return ccore_[u] >= k && in_ball(u); },
                         [](int64_t, int64_t) { return true; });
    }
    if (!opt.use_local) return kcore_component(q, k);
    return grow_region(q, g_, [&](int64_t u) { return core_[u] >= k && in_ball(u); },
                       [&](int64_t a, int64_t b) { return compatible(a, b); });
  }

  // Candidates for ballcore_r (any edge, ball of radius min(theta_v, theta_q) around q).
  std::vector<int64_t> candidates_r(int64_t q, int k, const Options& opt) {
    if (!opt.use_local) return kcore_component(q, k);
    const double tq = theta_[q];
    return grow_region(q, g_,
                       [&](int64_t u) {
                         return core_[u] >= k && dist(q, u) <= std::min(static_cast<double>(theta_[u]), tq);
                       },
                       [](int64_t, int64_t) { return true; });
  }

  // Peel alive (local mask) to the k-core and keep q's connected component.
  // Returns false if q does not survive. deg/queue are scratch buffers.
  bool peel_component(const Local& L, std::vector<char>& alive, int k, int32_t ql,
                      std::vector<int32_t>& deg, std::vector<int32_t>& buf) const {
    const int64_t s = L.size();
    deg.assign(static_cast<size_t>(s), 0);
    buf.clear();
    for (int64_t v = 0; v < s; ++v) {
      if (!alive[v]) continue;
      int32_t c = 0;
      for (int64_t e = L.off[v]; e < L.off[v + 1]; ++e) c += alive[L.adj[e]] ? 1 : 0;
      deg[v] = c;
      if (c < k) buf.push_back(static_cast<int32_t>(v));
    }
    for (size_t i = 0; i < buf.size(); ++i) {
      int32_t v = buf[i];
      if (!alive[v]) continue;
      alive[v] = 0;
      for (int64_t e = L.off[v]; e < L.off[v + 1]; ++e) {
        int32_t u = L.adj[e];
        if (alive[u]) {
          if (--deg[u] == k - 1) buf.push_back(u);
        }
      }
    }
    if (!alive[ql]) return false;
    // keep the component of q
    std::vector<char> mark(static_cast<size_t>(s), 0);
    buf.clear();
    buf.push_back(ql);
    mark[ql] = 1;
    for (size_t i = 0; i < buf.size(); ++i) {
      int32_t v = buf[i];
      for (int64_t e = L.off[v]; e < L.off[v + 1]; ++e) {
        int32_t u = L.adj[e];
        if (alive[u] && !mark[u]) {
          mark[u] = 1;
          buf.push_back(u);
        }
      }
    }
    for (int64_t v = 0; v < s; ++v)
      if (alive[v] && !mark[v]) alive[v] = 0;
    return true;
  }

  void finalize(Result& r, int64_t diam_cap) const {
    if (r.members.empty()) {
      r.feasible = false;
      if (r.status == "ok") r.status = "empty";
      return;
    }
    double mt = std::numeric_limits<double>::infinity();
    for (auto v : r.members) mt = std::min(mt, static_cast<double>(theta_[v]));
    r.min_theta = mt;
    double dm = 0.0;
    if (static_cast<int64_t>(r.members.size()) <= diam_cap) {
      for (size_t i = 0; i < r.members.size(); ++i)
        for (size_t j = i + 1; j < r.members.size(); ++j)
          dm = std::max(dm, dist(r.members[i], r.members[j]));
      r.diam = dm;
      r.feasible = dm <= mt + 1e-6;
    }
  }

  // ------------------------------------------------ latitude algorithms ----

  struct Interval {
    int64_t v;
    double l, r;
  };

  // For each tau in taus: count(tau) = #{v : l_v <= tau <= r_v}.
  static std::vector<int64_t> interval_counts(const std::vector<Interval>& iv, const std::vector<double>& taus) {
    std::vector<double> ls, rs;
    ls.reserve(iv.size());
    rs.reserve(iv.size());
    for (auto& x : iv) {
      ls.push_back(x.l);
      rs.push_back(x.r);
    }
    std::sort(ls.begin(), ls.end());
    std::sort(rs.begin(), rs.end());
    std::vector<int64_t> c(taus.size());
    for (size_t i = 0; i < taus.size(); ++i) {
      double t = taus[i];
      int64_t a = std::upper_bound(ls.begin(), ls.end(), t) - ls.begin();  // l <= t
      int64_t b = std::lower_bound(rs.begin(), rs.end(), t) - rs.begin();  // r < t
      c[i] = a - b;
    }
    return c;
  }

  // Candidates for query q: v in S0 with interval [scale * d(q,v), min(theta_v, theta_q)].
  std::vector<Interval> build_intervals(int64_t q, const std::vector<int64_t>& s0, double scale) const {
    std::vector<Interval> iv;
    const double tq = theta_[q];
    for (auto v : s0) {
      double r = std::min(static_cast<double>(theta_[v]), tq);
      double l = (v == q) ? 0.0 : scale * dist(q, v);
      if (l <= r) iv.push_back({v, l, r});
    }
    return iv;
  }

  // Per-threshold record of a sweep (v0.5), used by the bound pass of the exact algorithm.
  // kind: 0 not examined (bound = interval count), 1 bound = size of the peeled set C(tau),
  //       2 final (the threshold is resolved, or visit() returned a branch-and-bound root bound).
  struct SweepLog {
    std::vector<double> taus;
    std::vector<int64_t> cnt, bound;
    std::vector<int8_t> kind;
  };

  // Evaluate taus by decreasing interval count; for each, alive = {v : l<=tau<=r},
  // peel to k-core component of q. Calls visit(tau, survivors_global) and stops
  // when count <= stop_size() (if use_ub_order). visit returns an upper bound on
  // OPT_tau, or -1 if it has none (the peeled size is then used).
  void sweep(const Local& L, const std::vector<Interval>& iv, int32_t ql, const Options& opt, Result& r,
             const std::function<int64_t()>& stop_size,
             const std::function<int64_t(double, const std::vector<int64_t>&)>& visit, int max_eval = -1,
             SweepLog* log = nullptr) {
    std::vector<double> taus;
    for (auto& x : iv) taus.push_back(x.r);
    std::sort(taus.begin(), taus.end());
    taus.erase(std::unique(taus.begin(), taus.end()), taus.end());
    std::vector<int64_t> cnt = interval_counts(iv, taus);
    std::vector<size_t> order(taus.size());
    std::iota(order.begin(), order.end(), 0);
    if (opt.use_ub_order)
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cnt[a] > cnt[b]; });
    r.n_tau += static_cast<int64_t>(taus.size());
    if (log) {
      log->taus = taus;
      log->cnt = cnt;
      log->bound.assign(taus.size(), -1);
      log->kind.assign(taus.size(), 0);
    }
    auto note = [&](size_t ti, int64_t b, int8_t kind) {
      if (log) {
        log->bound[ti] = b;
        log->kind[ti] = kind;
      }
    };
    // local index of each interval's vertex (L.ids is in the same order as iv)
    std::vector<char> alive(static_cast<size_t>(L.size()));
    std::vector<int32_t> deg, buf;
    int evals = 0;
    for (size_t oi = 0; oi < order.size(); ++oi) {
      size_t ti = order[oi];
      if (cnt[ti] <= stop_size()) {
        if (opt.use_ub_order) {  // every later threshold has count <= cnt[ti]
          for (size_t oj = oi; oj < order.size(); ++oj) note(order[oj], cnt[order[oj]], 2);
          break;
        }
        note(ti, cnt[ti], 2);
        continue;
      }
      if (max_eval >= 0 && evals >= max_eval) break;
      if (Clock::now() > deadline_) {
        timed_out_ = true;
        break;
      }
      double tau = taus[ti];
      for (int64_t i = 0; i < L.size(); ++i) alive[i] = (iv[i].l <= tau && tau <= iv[i].r) ? 1 : 0;
      ++evals;
      r.n_tau_eval++;
      if (!peel_component(L, alive, opt.k, ql, deg, buf)) {
        note(ti, 0, 2);
        continue;
      }
      std::vector<int64_t> surv;
      for (int64_t i = 0; i < L.size(); ++i)
        if (alive[i]) surv.push_back(L.ids[i]);
      if (static_cast<int64_t>(surv.size()) <= stop_size()) {
        note(ti, static_cast<int64_t>(surv.size()), 2);
        continue;
      }
      const int64_t b = visit(tau, surv);
      if (b >= 0) note(ti, std::min<int64_t>(b, static_cast<int64_t>(surv.size())), 2);
      else note(ti, static_cast<int64_t>(surv.size()), 1);
    }
  }

  // Bound pass (v0.5): after an exact search that did not prove optimality, an upper bound on
  // OPT = max over thresholds of OPT_tau (Lemma 4). Thresholds are refined in decreasing order of
  // their current bound: interval count -> size of the peeled set C(tau) -> root bound of the
  // branch and bound on C(tau) (reductions with the incumbent, then the colouring bound), and the
  // pass stops as soon as the largest remaining bound is final or does not exceed the incumbent.
  // The pass has its own time budget; when it runs out, the largest remaining bound is reported.
  using Entry = std::tuple<int64_t, int, size_t>;  // (upper bound on OPT_tau, kind as in SweepLog, threshold)

  static std::vector<Entry> log_entries(const SweepLog& log) {
    std::vector<Entry> e;
    for (size_t ti = 0; ti < log.taus.size(); ++ti)
      e.push_back({log.kind[ti] == 0 ? log.cnt[ti] : log.bound[ti], log.kind[ti], ti});
    return e;
  }

  void bound_pass(const Local& L, const std::vector<Interval>& iv, int32_t ql, int64_t q, const Options& opt,
                  const std::vector<double>& taus, const std::vector<Entry>& entries, int64_t incumbent, Result& r) {
    const auto t0 = Clock::now();
    const double budget = opt.ub_time_s >= 0 ? opt.ub_time_s : std::max(5.0, 0.1 * opt.time_limit_s);
    const auto saved_deadline = deadline_;
    const bool saved_timeout = timed_out_;
    deadline_ = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(budget));
    timed_out_ = false;
    int64_t ub = incumbent;
    std::priority_queue<Entry> pq;  // (bound, kind, threshold)
    for (const auto& e : entries)
      if (std::get<0>(e) > ub) pq.push(e);
    r.n_tau_open = static_cast<int64_t>(pq.size());
    std::vector<char> alive(static_cast<size_t>(L.size()));
    std::vector<int32_t> deg, buf;
    while (!pq.empty()) {
      const auto [b, kind, ti] = pq.top();
      if (b <= ub) break;  // no threshold can beat the incumbent or the bound found so far
      pq.pop();
      if (kind >= 2 || Clock::now() > deadline_) {  // b is the largest bound left
        ub = b;
        break;
      }
      const double tau = taus[ti];
      for (int64_t i = 0; i < L.size(); ++i) alive[i] = (iv[i].l <= tau && tau <= iv[i].r) ? 1 : 0;
      if (!peel_component(L, alive, opt.k, ql, deg, buf)) continue;  // OPT_tau = 0
      std::vector<int64_t> surv;
      for (int64_t i = 0; i < L.size(); ++i)
        if (alive[i]) surv.push_back(L.ids[i]);
      const int64_t cs = static_cast<int64_t>(surv.size());
      if (cs <= ub) continue;
      if (kind == 0) {  // cheap refinement first: compare with the other bounds before colouring
        pq.push({cs, 1, ti});
        continue;
      }
      if (cs > opt.max_bound_size) {  // too large for a colouring bound
        pq.push({cs, 2, ti});
        continue;
      }
      std::vector<int64_t> sol;
      int64_t nodes = 0, rb = -1;
      branch_and_bound(q, surv, opt, ub, [&](int64_t a, int64_t c) { return dist(a, c) > tau; }, sol, nodes, &rb,
                       true);
      pq.push({rb >= 0 ? std::min(rb, cs) : cs, 2, ti});
    }
    r.ub = ub;
    r.ub_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    deadline_ = saved_deadline;
    timed_out_ = saved_timeout;
  }

  // Best-first LatSweep-Exact (v0.5). Every threshold carries an upper bound on OPT_tau that is
  // refined lazily, cheapest first: the interval count |S(tau)| -> the size of the peeled set C(tau)
  // -> the root bound of the branch and bound on C(tau) (reductions with the incumbent, then the
  // colouring bound) -> OPT_tau itself, by a full branch and bound. The threshold with the largest
  // bound is refined next, and the search stops once that bound does not exceed the incumbent, so a
  // full search runs only on thresholds whose root bound beats the incumbent, in decreasing order of
  // that bound (by Lemma 4 the incumbent is then optimal). The count order of v0.4 instead searches
  // thresholds by decreasing |S(tau)|, which can spend the time limit on thresholds whose optimum is
  // far below that of a later one. Returns the thresholds left open (with their current bounds) and
  // those too large for branch and bound; the caller decides the status and runs the bound pass.
  void exact_bestfirst(const Local& L, const std::vector<Interval>& iv, int32_t ql, int64_t q, const Options& opt,
                       std::vector<int64_t>& best, double& best_tau, Result& r, std::vector<double>& taus,
                       std::vector<Entry>& open, std::vector<Entry>& unresolved) {
    taus.clear();
    for (auto& x : iv) taus.push_back(x.r);
    std::sort(taus.begin(), taus.end());
    taus.erase(std::unique(taus.begin(), taus.end()), taus.end());
    const std::vector<int64_t> cnt = interval_counts(iv, taus);
    r.n_tau += static_cast<int64_t>(taus.size());
    auto inc = [&]() { return static_cast<int64_t>(best.size()); };
    std::priority_queue<Entry> pq;
    for (size_t ti = 0; ti < taus.size(); ++ti)
      if (cnt[ti] > inc()) pq.push({cnt[ti], 0, ti});
    std::vector<char> alive(static_cast<size_t>(L.size()));
    std::vector<int32_t> deg, buf;
    BBPrep prep;  // bitsets of the threshold bounded last; usually the next one searched
    size_t prep_ti = std::numeric_limits<size_t>::max();
    while (!pq.empty()) {
      const auto [key, kind, ti] = pq.top();
      if (key <= inc()) break;  // no threshold can beat the incumbent: optimal
      if (Clock::now() > deadline_) {
        timed_out_ = true;
        break;
      }
      pq.pop();
      const double tau = taus[ti];
      for (int64_t i = 0; i < L.size(); ++i) alive[i] = (iv[i].l <= tau && tau <= iv[i].r) ? 1 : 0;
      if (kind == 0) r.n_tau_eval++;
      if (!peel_component(L, alive, opt.k, ql, deg, buf)) continue;  // OPT_tau = 0
      std::vector<int64_t> surv;
      for (int64_t i = 0; i < L.size(); ++i)
        if (alive[i]) surv.push_back(L.ids[i]);
      const int64_t cs = static_cast<int64_t>(surv.size());
      if (cs <= inc()) continue;
      if (kind == 0) {
        pq.push({cs, 1, ti});
        continue;
      }
      auto conflict = [&](int64_t a, int64_t b) { return dist(a, b) > tau; };
      std::vector<int64_t> sol;
      int64_t nodes = 0, rb = -1;
      if (kind == 1) {
        if (cs > opt.max_bound_size) {  // no colouring bound for sets this large
          if (cs > opt.max_exact_size) unresolved.push_back({cs, 2, ti});
          else pq.push({cs, 2, ti});
          continue;
        }
        prep_ti = std::numeric_limits<size_t>::max();
        if (!bb_prepare(q, surv, conflict, opt.use_edge_filter, prep)) {  // time limit
          pq.push({cs, 1, ti});
          timed_out_ = true;
          break;
        }
        prep_ti = ti;
        const bool done = bb_search(prep, opt, inc(), sol, nodes, &rb, true);
        r.bb_nodes += nodes;
        if (sol.size() > best.size()) {
          best = sol;
          best_tau = tau;
        }
        if (done) continue;  // pruned or solved at the root: OPT_tau <= incumbent
        if (rb < 0) {        // time limit while building the conflict structure
          pq.push({cs, 1, ti});
          timed_out_ = true;
          break;
        }
        if (std::min(rb, cs) > inc()) pq.push({std::min(rb, cs), 2, ti});
        continue;
      }
      // kind 2: the full search
      if (cs > opt.max_exact_size) {
        unresolved.push_back({key, 2, ti});
        continue;
      }
      if (prep_ti != ti) {
        prep_ti = std::numeric_limits<size_t>::max();
        if (!bb_prepare(q, surv, conflict, opt.use_edge_filter, prep)) {  // time limit
          pq.push({key, 2, ti});
          timed_out_ = true;
          break;
        }
        prep_ti = ti;
      }
      const bool ok = bb_search(prep, opt, inc(), sol, nodes, &rb);
      r.bb_nodes += nodes;
      if (sol.size() > best.size()) {
        best = sol;
        best_tau = tau;
      }
      if (!ok) {  // time limit inside the search; keep the threshold open with its best known bound
        pq.push({rb >= 0 ? std::min(key, rb) : key, 2, ti});
        timed_out_ = true;
        break;
      }
    }
    while (!pq.empty()) {
      open.push_back(pq.top());
      pq.pop();
    }
  }

  void run_latitude(const std::string& algo, int64_t q, const Options& opt, Result& r) {
    const bool rtype = (algo == "ballcore_r");
    const double scale = (algo == "ballcore_f") ? 2.0 : 1.0;
    std::vector<int64_t> region = rtype ? candidates_r(q, opt.k, opt) : candidates_c(q, opt.k, scale, opt);
    r.s0_size = static_cast<int64_t>(region.size());
    const Graph& h = (!rtype && opt.use_index) ? hc_ : g_;
    std::vector<Interval> iv = build_intervals(q, region, scale);
    r.n_candidates = static_cast<int64_t>(iv.size());
    std::vector<int64_t> ids;
    ids.reserve(iv.size());
    for (auto& x : iv) ids.push_back(x.v);
    Local L = make_local(ids, h);
    int32_t ql = -1;
    for (int64_t i = 0; i < L.size(); ++i)
      if (L.ids[i] == q) ql = static_cast<int32_t>(i);
    if (ql < 0) {  // q cannot be in any feasible community (outside the k-core of G_theta)
      if (algo == "exact") r.status = "optimal";
      if (algo == "exact" || rtype) r.ub = 0;
      return;
    }

    std::vector<int64_t> best;
    double best_tau = std::numeric_limits<double>::quiet_NaN();
    auto best_size = [&]() { return static_cast<int64_t>(best.size()); };

    if (algo == "ballcore_f" || algo == "ballcore_r") {
      sweep(L, iv, ql, opt, r, best_size, [&](double tau, const std::vector<int64_t>& surv) -> int64_t {
        if (surv.size() > best.size()) {
          best = surv;
          best_tau = tau;
        }
        return -1;
      });
      r.members = best;
      r.tau = best_tau;
      if (timed_out_) r.status = "timeout";
      else if (rtype) r.ub = static_cast<int64_t>(best.size());  // |B| >= OPT once the sweep has finished
      return;
    }

    if (algo == "greedy") {
      sweep(L, iv, ql, opt, r, best_size,
            [&](double tau, const std::vector<int64_t>& surv) -> int64_t {
              std::vector<int64_t> sol = greedy_peel(q, surv, opt.k, [&](int64_t a, int64_t b) {
                return dist(a, b) > tau;
              });
              if (sol.size() > best.size()) {
                best = sol;
                best_tau = tau;
              }
              return -1;
            },
            opt.greedy_max_tau);
      r.members = best;
      r.tau = best_tau;
      if (timed_out_) r.status = "timeout";
      return;
    }

    // exact: seed with feasible solutions, then branch-and-bound per tau
    if (opt.use_seed) {
      Options o2 = opt;
      o2.use_ub_order = true;  // v0.5: the seed is the same whatever the main sweep's order
      Result tmp;
      std::vector<Interval> ivf = build_intervals(q, candidates_c(q, opt.k, 2.0, opt), 2.0);
      std::vector<int64_t> idf;
      for (auto& x : ivf) idf.push_back(x.v);
      Local Lf = make_local(idf, h);
      int32_t qf = -1;
      for (int64_t i = 0; i < Lf.size(); ++i)
        if (Lf.ids[i] == q) qf = static_cast<int32_t>(i);
      if (qf >= 0)
        sweep(Lf, ivf, qf, o2, tmp, best_size, [&](double tau, const std::vector<int64_t>& surv) -> int64_t {
          if (surv.size() > best.size()) {
            best = surv;
            best_tau = tau;
          }
          return -1;
        });
      sweep(L, iv, ql, o2, tmp, best_size,
            [&](double tau, const std::vector<int64_t>& surv) -> int64_t {
              std::vector<int64_t> sol = greedy_peel(q, surv, opt.k, [&](int64_t a, int64_t b) {
                return dist(a, b) > tau;
              });
              if (sol.size() > best.size()) {
                best = sol;
                best_tau = tau;
              }
              return -1;
            },
            opt.greedy_max_tau);
    }
    greedy_skipped_ = false;  // seeding is optional; skipping it does not affect optimality
    if (opt.tau_order_bound && opt.use_ub_order) {
      std::vector<double> taus;
      std::vector<Entry> open, unresolved;
      exact_bestfirst(L, iv, ql, q, opt, best, best_tau, r, taus, open, unresolved);
      r.members = best;
      r.tau = best_tau;
      const int64_t inc = static_cast<int64_t>(best.size());
      bool large_open = false;  // a threshold too large for branch and bound whose bound beats the incumbent
      for (const auto& e : unresolved) large_open = large_open || std::get<0>(e) > inc;
      if (timed_out_) r.status = "timeout";
      else if (large_open) r.status = "too_large";
      else r.status = "optimal";
      if (r.status == "optimal") {
        r.ub = inc;
      } else if (opt.compute_ub) {
        open.insert(open.end(), unresolved.begin(), unresolved.end());
        bound_pass(L, iv, ql, q, opt, taus, open, inc, r);
      }
      return;
    }
    bool too_large = false;
    SweepLog log;
    sweep(
        L, iv, ql, opt, r, best_size,
        [&](double tau, const std::vector<int64_t>& surv) -> int64_t {
          if (timed_out_) return -1;
          if (static_cast<int64_t>(surv.size()) > opt.max_exact_size) {
            // too large for branch-and-bound: the answer keeps the seeded incumbent (BallCore-F and greedy
            // already ran) and the status reports that optimality is not proven
            too_large = true;
            return -1;
          }
          std::vector<int64_t> sol;
          int64_t nodes = 0, rb = -1;
          const int64_t lb = static_cast<int64_t>(best.size());
          bool ok = branch_and_bound(q, surv, opt, lb, [&](int64_t a, int64_t b) { return dist(a, b) > tau; }, sol,
                                     nodes, &rb);
          r.bb_nodes += nodes;
          if (!ok) timed_out_ = true;
          if (sol.size() > best.size()) {
            best = sol;
            best_tau = tau;
          }
          if (ok) return std::max<int64_t>(lb, static_cast<int64_t>(sol.size()));  // OPT_tau is settled
          return rb;  // root bound of the interrupted search (-1 if it never got past the root)
        },
        -1, &log);
    r.members = best;
    r.tau = best_tau;
    if (timed_out_) r.status = "timeout";
    else if (too_large) r.status = "too_large";
    else r.status = "optimal";
    if (r.status == "optimal") r.ub = static_cast<int64_t>(best.size());
    else if (opt.compute_ub) bound_pass(L, iv, ql, q, opt, log.taus, log_entries(log), static_cast<int64_t>(best.size()), r);
  }

  // ------------------------------------------ fixed-threshold algorithms ----

  void run_threshold(const std::string& algo, int64_t q, const Options& opt, Result& r) {
    const bool kr = (algo.rfind("kr_", 0) == 0);
    if (kr && opt.r < 0) throw std::runtime_error("kr_* algorithms need --r");
    const int k = opt.k;
    std::vector<int64_t> region;
    if (kr) {
      region = opt.use_local
                   ? grow_region(q, g_, [&](int64_t u) { return core_[u] >= k && dist(q, u) <= opt.r; },
                                 [&](int64_t a, int64_t b) { return dist(a, b) <= opt.r; })
                   : kcore_component(q, k);
    } else {
      region = candidates_c(q, k, 1.0, opt);
    }
    r.s0_size = static_cast<int64_t>(region.size());
    std::vector<int64_t> cand;
    for (auto v : region) {
      double d = (v == q) ? 0.0 : dist(q, v);
      double lim = kr ? opt.r : std::min(theta_[q], theta_[v]);
      if (d <= lim) cand.push_back(v);
    }
    r.n_candidates = static_cast<int64_t>(cand.size());
    Local L = make_local(cand, (!kr && opt.use_index) ? hc_ : g_);
    int32_t ql = -1;
    for (int64_t i = 0; i < L.size(); ++i)
      if (L.ids[i] == q) ql = static_cast<int32_t>(i);
    const bool is_exact = algo.size() > 6 && algo.compare(algo.size() - 6, 6, "_exact") == 0;
    if (ql < 0) {
      r.status = "empty";
      if (is_exact) r.ub = 0;
      return;
    }
    std::vector<char> alive(static_cast<size_t>(L.size()), 1);
    std::vector<int32_t> deg, buf;
    if (!peel_component(L, alive, opt.k, ql, deg, buf)) {
      r.status = "empty";
      if (is_exact) r.ub = 0;
      return;
    }
    std::vector<int64_t> surv;
    for (int64_t i = 0; i < L.size(); ++i)
      if (alive[i]) surv.push_back(L.ids[i]);
    auto conflict = [&](int64_t a, int64_t b) {
      double d = dist(a, b);
      double lim = kr ? opt.r : std::min(theta_[a], theta_[b]);
      return d > lim;
    };
    r.tau = kr ? opt.r : std::numeric_limits<double>::quiet_NaN();
    const bool exact = algo.size() > 6 && algo.compare(algo.size() - 6, 6, "_exact") == 0;
    std::vector<int64_t> best = greedy_peel(q, surv, opt.k, conflict);
    if (!exact) {
      r.members = best;
      return;
    }
    if (static_cast<int64_t>(surv.size()) > opt.max_exact_size) {
      r.members = best;
      r.status = "too_large";
      r.ub = static_cast<int64_t>(surv.size());
      if (opt.compute_ub && static_cast<int64_t>(surv.size()) <= opt.max_bound_size) {
        const auto t0 = Clock::now();
        const double budget = opt.ub_time_s >= 0 ? opt.ub_time_s : std::max(5.0, 0.1 * opt.time_limit_s);
        deadline_ = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(budget));
        std::vector<int64_t> sol;
        int64_t nodes = 0, rb = -1;
        branch_and_bound(q, surv, opt, static_cast<int64_t>(best.size()), conflict, sol, nodes, &rb, true);
        if (rb >= 0) r.ub = std::min(r.ub, rb);
        r.ub_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      }
      return;
    }
    greedy_skipped_ = false;
    std::vector<int64_t> sol;
    int64_t nodes = 0, rb = -1;
    const int64_t lb = opt.use_seed ? static_cast<int64_t>(best.size()) : 0;
    bool ok = branch_and_bound(q, surv, opt, lb, conflict, sol, nodes, &rb);
    r.bb_nodes = nodes;
    if (sol.size() > best.size()) best = sol;
    r.members = best;
    r.status = ok ? "optimal" : "timeout";
    if (ok) r.ub = static_cast<int64_t>(best.size());
    else r.ub = std::max<int64_t>(static_cast<int64_t>(best.size()), rb >= 0 ? rb : static_cast<int64_t>(surv.size()));
  }

  // ----------------------------------------------------- spatial baseline ----

  // SAC search (Fang et al., PVLDB 2017), AppInc: the connected k-core containing q inside
  // the smallest ball centred at q that holds one, i.e. a 2-approximation of the community
  // with the minimum covering circle. Uses positions only, no tolerances; tau = the radius.
  // Feasibility is monotone in the radius (a k-core inside a ball stays one in any larger
  // ball, and q's k-core component can only grow), so the radius is found by binary search
  // over the sorted distances to q. opt.min_size > 0 also asks for at least that many members.
  void run_sac(int64_t q, const Options& opt, Result& r) {
    std::vector<int64_t> comp = kcore_component(q, opt.k);
    r.s0_size = static_cast<int64_t>(comp.size());
    std::vector<std::pair<double, int64_t>> byd;
    byd.reserve(comp.size());
    for (auto v : comp) byd.push_back({v == q ? 0.0 : dist(q, v), v});
    std::sort(byd.begin(), byd.end());
    const int64_t n = static_cast<int64_t>(byd.size());
    r.n_candidates = n;
    auto prefix_end = [&](int64_t p) {  // the p nearest vertices, plus ties at the same distance
      int64_t e = p;
      while (e < n && byd[static_cast<size_t>(e)].first <= byd[static_cast<size_t>(p - 1)].first) ++e;
      return e;
    };
    std::vector<int32_t> deg, buf;
    auto core_in = [&](int64_t e, std::vector<int64_t>* out) -> bool {
      std::vector<int64_t> ids;
      ids.reserve(static_cast<size_t>(e));
      for (int64_t i = 0; i < e; ++i) ids.push_back(byd[static_cast<size_t>(i)].second);
      Local L = make_local(ids);
      int32_t ql = -1;
      for (int64_t i = 0; i < L.size(); ++i)
        if (L.ids[i] == q) {
          ql = static_cast<int32_t>(i);
          break;
        }
      if (ql < 0) return false;
      std::vector<char> alive(static_cast<size_t>(L.size()), 1);
      if (!peel_component(L, alive, opt.k, ql, deg, buf)) return false;
      int64_t cnt = 0;
      for (int64_t i = 0; i < L.size(); ++i) cnt += alive[i] ? 1 : 0;
      if (out) {
        out->clear();
        for (int64_t i = 0; i < L.size(); ++i)
          if (alive[i]) out->push_back(L.ids[i]);
      }
      return cnt >= opt.min_size || e == n;  // the whole component is the fallback when it is too small
    };
    // invariant: the ball of the lo nearest vertices holds no k-core with q; that of the hi nearest does
    // (all of q's k-core component is one)
    int64_t lo = 0, hi = n;
    while (hi - lo > 1) {
      if (Clock::now() > deadline_) {
        timed_out_ = true;
        break;
      }
      const int64_t mid = lo + (hi - lo) / 2;
      if (core_in(prefix_end(mid), nullptr)) hi = mid;
      else lo = mid;
    }
    const int64_t e = prefix_end(hi);
    core_in(e, &r.members);
    r.tau = byd[static_cast<size_t>(e - 1)].first;
    if (timed_out_) r.status = "timeout";
  }

  void run_edgeacc(int64_t q, const Options& opt, Result& r) {
    if (opt.use_index) {  // the k-core component of q in G_theta, read off the index
      r.members = component_on(hc_, ccore_, q, opt.k);
      r.n_candidates = static_cast<int64_t>(r.members.size());
      return;
    }
    std::vector<int64_t> s0 = kcore_component(q, opt.k);
    r.s0_size = static_cast<int64_t>(s0.size());
    Local L = make_local(s0);
    r.n_candidates = L.size();
    // filter local edges by mutual acceptance
    Local F;
    F.ids = L.ids;
    F.off.assign(static_cast<size_t>(L.size() + 1), 0);
    std::vector<int32_t> adj;
    adj.reserve(L.adj.size());
    for (int64_t v = 0; v < L.size(); ++v) {
      for (int64_t e = L.off[v]; e < L.off[v + 1]; ++e) {
        int32_t u = L.adj[e];
        int64_t gv = L.ids[v], gu = L.ids[u];
        double lim = std::min(theta_[gv], theta_[gu]);
        if (dist(gv, gu) <= lim) adj.push_back(u);
      }
      F.off[v + 1] = static_cast<int64_t>(adj.size());
    }
    F.adj = std::move(adj);
    int32_t ql = -1;
    for (int64_t i = 0; i < F.size(); ++i)
      if (F.ids[i] == q) ql = static_cast<int32_t>(i);
    std::vector<char> alive(static_cast<size_t>(F.size()), 1);
    std::vector<int32_t> deg, buf;
    if (!peel_component(F, alive, opt.k, ql, deg, buf)) return;
    for (int64_t i = 0; i < F.size(); ++i)
      if (alive[i]) r.members.push_back(F.ids[i]);
  }

  // --------------------------------------------- conflict structure ----

  // Conflict bitsets over the local candidates, and the local graph with every
  // edge between two conflicting vertices removed: such an edge can never lie
  // inside a feasible community, so peeling on this filtered graph is valid
  // and prunes much harder than peeling on G.
  template <class Conflict>
  bool conflict_structure(const Local& L, Conflict conflict, bool filter_edges, std::vector<Bitset>& conf,
                          Local& F) const {
    const int64_t s = L.size();
    conf.assign(static_cast<size_t>(s), Bitset(s));
    for (int64_t i = 0; i < s; ++i) {
      if ((i & 255) == 255 && Clock::now() > deadline_) return false;
      for (int64_t j = i + 1; j < s; ++j)
        if (conflict(L.ids[i], L.ids[j])) {
          conf[i].set(j);
          conf[j].set(i);
        }
    }
    F.ids = L.ids;
    F.off.assign(static_cast<size_t>(s + 1), 0);
    F.adj.clear();
    F.adj.reserve(L.adj.size());
    for (int64_t v = 0; v < s; ++v) {
      for (int64_t e = L.off[v]; e < L.off[v + 1]; ++e) {
        int32_t u = L.adj[e];
        if (!filter_edges || !conf[v].test(u)) F.adj.push_back(u);
      }
      F.off[v + 1] = static_cast<int64_t>(F.adj.size());
    }
    return true;
  }

  // ------------------------------------------------------------ greedy ----

  // Repeatedly drop the vertex (other than q) with most conflicts, re-peel,
  // until no conflicts remain. Always returns a feasible set or empty.
  template <class Conflict>
  std::vector<int64_t> greedy_peel(int64_t q, const std::vector<int64_t>& cand, int k, Conflict conflict) {
    if (static_cast<int64_t>(cand.size()) > greedy_cap_) {
      greedy_skipped_ = true;
      return {};
    }
    Local L = make_local(cand);
    const int64_t s = L.size();
    int32_t ql = -1;
    for (int64_t i = 0; i < s; ++i)
      if (L.ids[i] == q) ql = static_cast<int32_t>(i);
    if (ql < 0) return {};
    std::vector<Bitset> conf;
    Local F;
    if (!conflict_structure(L, conflict, true, conf, F)) {
      timed_out_ = true;
      return {};
    }
    std::vector<char> alive(static_cast<size_t>(s), 1), before;
    std::vector<int32_t> deg, buf;
    if (!peel_component(F, alive, k, ql, deg, buf)) return {};
    Bitset aliveB(s);
    for (int64_t i = 0; i < s; ++i)
      if (alive[i]) aliveB.set(i);
    std::vector<int64_t> ccount(static_cast<size_t>(s), 0);
    int64_t total = 0;
    for (int64_t i = 0; i < s; ++i)
      if (alive[i]) {
        ccount[i] = and_count(conf[i], aliveB);
        total += ccount[i];
      }
    auto bury = [&](int64_t v) {  // v just died: update partners' counts
      aliveB.reset(v);
      conf[v].for_each([&](int64_t u) {
        if (aliveB.test(u)) {
          --ccount[u];
          total -= 2;
        }
      });
      ccount[v] = 0;
    };
    int64_t iters = 0;
    while (total > 0) {
      if ((++iters & 63) == 0 && Clock::now() > deadline_) {
        timed_out_ = true;
        return {};
      }
      int64_t pick = -1;
      for (int64_t i = 0; i < s; ++i) {
        if (!alive[i] || i == ql || ccount[i] == 0) continue;
        if (pick < 0 || ccount[i] > ccount[pick] ||
            (ccount[i] == ccount[pick] && (F.off[i + 1] - F.off[i]) < (F.off[pick + 1] - F.off[pick])))
          pick = i;
      }
      if (pick < 0) return {};
      before = alive;
      alive[pick] = 0;
      if (!peel_component(F, alive, k, ql, deg, buf)) return {};
      for (int64_t i = 0; i < s; ++i)
        if (before[i] && !alive[i]) bury(i);
    }
    std::vector<int64_t> out;
    for (int64_t i = 0; i < s; ++i)
      if (alive[i]) out.push_back(L.ids[i]);
    return out;
  }

  // --------------------------------------------------- branch and bound ----

  // Maximum H subset of cand with q in H, G[H] connected k-core, no conflicting
  // pair. Returns false on timeout. Only solutions strictly larger than
  // lower_bound are reported. Pruning rules (all sound):
  //   * peeling on the conflict-filtered graph (k-core + connectivity to q);
  //   * compatibility degree: a member of a solution larger than the incumbent
  //     must be compatible with at least |incumbent| other members;
  //   * greedy coloring of the compatibility graph (clique-size bound).
  //
  // v0.5: *root_ub (if given) receives an upper bound on the largest feasible subset, valid
  // whether or not the search completes: the bound of the root node after rules (1)-(3), or
  // lower_bound when the root is pruned; -1 if the root was not reached (time limit while
  // building the conflict structure). With bound_only the search stops after the root.
  template <class Conflict>
  bool branch_and_bound(int64_t q, const std::vector<int64_t>& cand, const Options& opt, int64_t lower_bound,
                        Conflict conflict, std::vector<int64_t>& best_out, int64_t& nodes,
                        int64_t* root_ub = nullptr, bool bound_only = false) {
    BBPrep P;
    if (root_ub) *root_ub = -1;
    if (!bb_prepare(q, cand, conflict, opt.use_edge_filter, P)) return false;
    return bb_search(P, opt, lower_bound, best_out, nodes, root_ub, bound_only);
  }

  // The data a branch and bound needs for one candidate set (v0.5: kept separately so that the
  // best-first sweep can bound a threshold and later search it without rebuilding the bitsets).
  struct BBPrep {
    Local L, F;                 // local graph over the candidates, and the same without conflicting edges
    std::vector<Bitset> conf;   // conflict bitsets
    int32_t ql = -1;            // local index of q (-1: q is not a candidate)
  };

  // Returns false at the time limit (while building the conflict bitsets).
  template <class Conflict>
  bool bb_prepare(int64_t q, const std::vector<int64_t>& cand, Conflict conflict, bool filter_edges, BBPrep& P) {
    P.L = make_local(cand);
    P.F = Local();
    P.conf.clear();
    P.ql = -1;
    for (int64_t i = 0; i < P.L.size(); ++i)
      if (P.L.ids[i] == q) P.ql = static_cast<int32_t>(i);
    if (P.ql < 0) return true;
    return conflict_structure(P.L, conflict, filter_edges, P.conf, P.F);
  }

  bool bb_search(const BBPrep& P, const Options& opt, int64_t lower_bound, std::vector<int64_t>& best_out,
                 int64_t& nodes, int64_t* root_ub = nullptr, bool bound_only = false) {
    const Local& L = P.L;
    const Local& F = P.F;
    const std::vector<Bitset>& conf = P.conf;
    const int64_t s = L.size();
    const int32_t ql = P.ql;
    if (root_ub) *root_ub = -1;
    if (ql < 0) {
      if (root_ub) *root_ub = lower_bound;
      return true;
    }
    int64_t best_size = lower_bound;
    Bitset best_set(s);
    bool found = false;
    bool aborted = false;
    std::vector<char> alive(static_cast<size_t>(s));
    std::vector<int32_t> deg, buf;
    Bitset aliveB(s);

    bool at_root = true;
    // the root's bound, recorded once: lower_bound if the root is pruned
    auto root_bound = [&](int64_t b) {
      if (at_root) {
        if (root_ub) *root_ub = std::max(b, lower_bound);
        at_root = false;
      }
    };
    std::function<void(Bitset&, Bitset&)> rec = [&](Bitset& M, Bitset& C) {
      if (aborted) return;
      ++nodes;
      // v0.5: the time limit is checked at every node and inside the colouring below (v0.4: every 256
      // nodes), since one node costs up to O(s^2 * s/64) on large candidate sets
      if (Clock::now() > deadline_) {
        aborted = true;
        return;
      }
      for (int64_t i = 0; i < s; ++i) alive[i] = (M.test(i) || C.test(i)) ? 1 : 0;
      for (int pass = 0;; ++pass) {
        if (pass > 0 && Clock::now() > deadline_) {
          aborted = true;
          return;
        }
        if (!peel_component(F, alive, opt.k, ql, deg, buf)) return root_bound(lower_bound);
        bool m_ok = true;
        M.for_each([&](int64_t i) {
          if (!alive[i]) m_ok = false;
        });
        if (!m_ok) return root_bound(lower_bound);
        if (!opt.use_compat_bound || best_size <= 0) break;
        std::fill(aliveB.w.begin(), aliveB.w.end(), 0ULL);
        int64_t na = 0;
        for (int64_t i = 0; i < s; ++i)
          if (alive[i]) {
            aliveB.set(i);
            ++na;
          }
        if (na <= best_size) return root_bound(lower_bound);
        bool removed = false;
        for (int64_t i = 0; i < s; ++i) {
          if (!alive[i]) continue;
          int64_t compat = na - 1 - and_count(conf[i], aliveB);
          if (compat < best_size) {
            if (M.test(i)) return root_bound(lower_bound);
            alive[i] = 0;
            removed = true;
          }
        }
        if (!removed) break;
      }
      Bitset C2(s);
      int64_t csize = 0;
      for (int64_t i = 0; i < s; ++i)
        if (alive[i] && !M.test(i)) {
          C2.set(i);
          ++csize;
        }
      const int64_t msize = M.count();
      if (msize + csize <= best_size) return root_bound(lower_bound);
      int64_t pick = -1, pick_c = 0;
      bool any_conf = false;
      C2.for_each([&](int64_t i) {
        int64_t c = and_count(conf[i], C2);
        if (c > 0) any_conf = true;
        if (c > pick_c || (c == pick_c && c > 0 && pick >= 0 &&
                           (F.off[i + 1] - F.off[i]) < (F.off[pick + 1] - F.off[pick]))) {
          pick = i;
          pick_c = c;
        }
      });
      if (!any_conf) {
        root_bound(msize + csize);
        best_size = msize + csize;
        best_set = M;
        C2.for_each([&](int64_t i) { best_set.set(i); });
        found = true;
        return;
      }
      if (opt.use_color_bound) {
        // greedy coloring of the compatibility graph on C2: each color class is a
        // set of pairwise-conflicting vertices, so a feasible subset uses at most
        // one vertex per class.
        // v0.5: colour in order of decreasing compatibility degree within C2 (fewest conflicts first),
        // as in Tomita-Seki; this gives fewer colour classes than index order (the v0.4 order).
        std::vector<std::pair<int64_t, int64_t>> ord;
        C2.for_each([&](int64_t i) { ord.push_back({opt.color_order_degree ? and_count(conf[i], C2) : 0, i}); });
        if (opt.color_order_degree) std::sort(ord.begin(), ord.end());
        std::vector<Bitset> classes;
        int64_t coloured = 0;
        for (const auto& pr : ord) {
          if ((++coloured & 63) == 0 && Clock::now() > deadline_) {
            aborted = true;
            return;
          }
          const int64_t i = pr.second;
          bool placed = false;
          for (auto& cls : classes) {
            bool ok = true;  // i can join cls iff it conflicts with every member
            for (size_t w = 0; w < cls.w.size() && ok; ++w)
              if (cls.w[w] & ~conf[i].w[w]) ok = false;
            if (ok) {
              cls.set(i);
              placed = true;
              break;
            }
          }
          if (!placed) {
            classes.emplace_back(s);
            classes.back().set(i);
          }
        }
        const int64_t cb = msize + static_cast<int64_t>(classes.size());
        root_bound(cb);
        if (cb <= best_size) return;
      } else {
        root_bound(msize + csize);
      }
      if (bound_only) {
        aborted = true;  // stop after the root: the caller only wants its bound
        return;
      }
      {  // branch 1: exclude pick
        Bitset M1 = M;
        Bitset C1 = C2;
        C1.reset(pick);
        rec(M1, C1);
      }
      if (aborted) return;
      {  // branch 2: include pick, drop its conflicts
        Bitset M2 = M;
        M2.set(pick);
        Bitset Cn = C2;
        Cn.reset(pick);
        for (size_t w = 0; w < Cn.w.size(); ++w) Cn.w[w] &= ~conf[pick].w[w];
        rec(M2, Cn);
      }
    };

    Bitset M(s), C(s);
    M.set(ql);
    for (int64_t i = 0; i < s; ++i)
      if (i != ql) C.set(i);
    rec(M, C);
    if (found) {
      best_out.clear();
      best_set.for_each([&](int64_t i) { best_out.push_back(L.ids[i]); });
    }
    return !aborted;
  }
};

}  // namespace lacs
