// lacs_lpa: size-capped asynchronous label propagation on a LACSGR01 graph.
//
//   lacs_lpa --graph graph.bin --out labels.bin [--max-size 1000] [--iters 20] [--seed 0] [--tol 0.001]
//
// Used by E3 to give large graphs community-aligned synthetic attributes. Each vertex
// starts in its own community; in random order every vertex adopts the most frequent
// label among its neighbours (ties broken at random), skipping labels that already have
// --max-size members (its own label is always allowed). Stops after --iters sweeps or
// when fewer than tol * n vertices change. Output: int32 labels 0..C-1 (raw, n values);
// a JSON line with summary statistics goes to stdout.
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>

#include "lacs_core.hpp"

int main(int argc, char** argv) {
  std::string graph_path, out_path;
  int64_t max_size = 1000;
  int iters = 20;
  uint64_t seed = 0;
  double tol = 0.001;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--graph") graph_path = next();
    else if (k == "--out") out_path = next();
    else if (k == "--max-size") max_size = std::stoll(next());
    else if (k == "--iters") iters = std::stoi(next());
    else if (k == "--seed") seed = std::stoull(next());
    else if (k == "--tol") tol = std::stod(next());
    else {
      std::cerr << "usage: lacs_lpa --graph graph.bin --out labels.bin [--max-size 1000] [--iters 20] [--seed 0]\n";
      return 2;
    }
  }
  if (graph_path.empty() || out_path.empty()) {
    std::cerr << "usage: lacs_lpa --graph graph.bin --out labels.bin [--max-size 1000] [--iters 20] [--seed 0]\n";
    return 2;
  }
  try {
    lacs::Graph g = lacs::read_graph(graph_path);
    const int64_t n = g.n;
    std::vector<int32_t> label(static_cast<size_t>(n));
    std::iota(label.begin(), label.end(), 0);
    std::vector<int64_t> size(static_cast<size_t>(n), 1);
    std::vector<int32_t> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(seed);
    std::vector<int32_t> buf;
    int done = 0;
    int64_t last_changes = 0;
    for (int it = 0; it < iters; ++it) {
      std::shuffle(order.begin(), order.end(), rng);
      int64_t changes = 0;
      for (int32_t v : order) {
        const int64_t b = g.off[v], e = g.off[v + 1];
        if (b == e) continue;
        buf.clear();
        for (int64_t p = b; p < e; ++p) buf.push_back(label[g.adj[p]]);
        std::sort(buf.begin(), buf.end());
        const int32_t own = label[v];
        int32_t best = own;
        int64_t best_c = -1, ties = 0;
        for (size_t i = 0; i < buf.size();) {
          size_t j = i;
          while (j < buf.size() && buf[j] == buf[i]) ++j;
          const int32_t lab = buf[i];
          const int64_t c = static_cast<int64_t>(j - i);
          if (lab == own || size[lab] < max_size) {
            if (c > best_c) {
              best = lab;
              best_c = c;
              ties = 1;
            } else if (c == best_c) {  // reservoir sampling over tied labels
              ++ties;
              if (std::uniform_int_distribution<int64_t>(1, ties)(rng) == 1) best = lab;
            }
          }
          i = j;
        }
        if (best != own) {
          --size[own];
          ++size[best];
          label[v] = best;
          ++changes;
        }
      }
      done = it + 1;
      last_changes = changes;
      if (static_cast<double>(changes) < tol * static_cast<double>(n)) break;
    }
    // compact labels to 0..C-1
    std::vector<int32_t> map(static_cast<size_t>(n), -1);
    int32_t C = 0;
    for (int64_t v = 0; v < n; ++v) {
      int32_t& m = map[label[v]];
      if (m < 0) m = C++;
      label[v] = m;
    }
    std::vector<int64_t> csize(static_cast<size_t>(C), 0);
    for (auto l : label) ++csize[l];
    std::vector<int64_t> sorted = csize;
    std::sort(sorted.begin(), sorted.end());
    auto q = [&](double f) { return sorted.empty() ? 0 : sorted[static_cast<size_t>(f * (sorted.size() - 1))]; };
    int64_t in20 = 0;
    for (auto c : csize)
      if (c >= 20) in20 += c;
    FILE* f = std::fopen(out_path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + out_path);
    std::fwrite(label.data(), sizeof(int32_t), label.size(), f);
    std::fclose(f);
    std::cout << "{\"n\":" << n << ",\"communities\":" << C << ",\"iterations\":" << done
              << ",\"last_changes\":" << last_changes << ",\"size_p50\":" << q(0.5) << ",\"size_p90\":" << q(0.9)
              << ",\"size_max\":" << (sorted.empty() ? 0 : sorted.back())
              << ",\"vertices_in_communities_ge20\":" << in20 << "}" << std::endl;
  } catch (const std::exception& ex) {
    std::cerr << "lacs_lpa: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
