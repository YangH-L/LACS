// lacs_convert: edge list text -> LACSGR01 binary CSR graph.
//
//   lacs_convert --in edges.txt --out graph.bin [--relabel ids.txt] [--n N] [--lowmem]
//
// Without --relabel the ids must be 0..n-1 (n = max id + 1 unless --n is given).
// With --relabel, arbitrary integer ids are mapped to 0..n-1 in sorted order and
// the original id of each new vertex is written to ids.txt (one per line).
//
// --lowmem (v0.5) reads the file twice instead of holding all pairs in memory: the first
// pass counts degrees per id, the second fills the adjacency lists, which are then sorted
// and deduplicated in place. Peak memory is about 4 bytes per adjacency entry (2m entries,
// before deduplication) plus 13 bytes per id up to the largest id, instead of about 40 bytes
// per edge, so a graph with 1.8 billion edges fits in about 16 GB. The output is the same
// file as without --lowmem (same relabelling, sorted adjacency, no self loops or duplicate
// edges). It needs non-negative ids below 2^31.
#include <algorithm>
#include <iostream>

#include "edge_reader.hpp"
#include "lacs_core.hpp"

namespace {

lacs::Graph convert_lowmem(const std::string& in, const std::string& relabel, int64_t n_given) {
  // pass 1: degree per original id (self loops excluded; their endpoints still count as present)
  std::vector<uint32_t> deg;  // later reused as the id -> new id map
  std::vector<uint8_t> present;
  int64_t max_id = -1, pairs = 0;
  auto grow = [&](int64_t id) {
    if (id < 0) throw std::runtime_error("negative id; --lowmem needs non-negative ids");
    if (id >= (int64_t{1} << 31) - 1) throw std::runtime_error("id >= 2^31 - 1; --lowmem cannot store it");
    if (id >= static_cast<int64_t>(deg.size())) {
      size_t sz = std::max<size_t>(static_cast<size_t>(id) + 1, deg.size() * 3 / 2 + 1024);
      deg.resize(sz, 0);
      present.resize(sz, 0);
    }
    max_id = std::max(max_id, id);
  };
  lacs::for_each_int_pair(in, [&](int64_t a, int64_t b) {
    grow(a);
    grow(b);
    present[a] = 1;
    present[b] = 1;
    ++pairs;
    if (a != b) {
      if (deg[a] == UINT32_MAX || deg[b] == UINT32_MAX) throw std::runtime_error("degree overflow");
      ++deg[a];
      ++deg[b];
    }
  });
  const int64_t ids_n = max_id + 1;
  deg.resize(static_cast<size_t>(ids_n));
  present.resize(static_cast<size_t>(ids_n));
  deg.shrink_to_fit();
  present.shrink_to_fit();
  lacs::Graph g;
  int64_t n = 0;
  if (!relabel.empty()) {
    for (int64_t id = 0; id < ids_n; ++id) n += present[id];
  } else {
    n = (n_given >= 0) ? n_given : ids_n;
    if (ids_n > n) throw std::runtime_error("edge endpoint out of range");
  }
  if (n >= (int64_t{1} << 31)) throw std::runtime_error("more than 2^31 - 1 vertices");
  g.n = n;
  g.off.assign(static_cast<size_t>(n + 1), 0);
  if (!relabel.empty()) {
    FILE* f = std::fopen(relabel.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + relabel);
    uint32_t r = 0;
    for (int64_t id = 0; id < ids_n; ++id) {
      if (!present[id]) continue;
      std::fprintf(f, "%lld\n", static_cast<long long>(id));
      g.off[r + 1] = g.off[r] + deg[id];
      deg[id] = r++;  // id -> new id
    }
    std::fclose(f);
  } else {
    for (int64_t v = 0; v < n; ++v) g.off[v + 1] = g.off[v] + (v < ids_n ? deg[v] : 0);
    for (int64_t id = 0; id < ids_n; ++id) deg[id] = static_cast<uint32_t>(id);
  }
  std::vector<uint8_t>().swap(present);
  // pass 2: fill
  g.adj.resize(static_cast<size_t>(g.off[n]));
  std::vector<int64_t> cur(g.off.begin(), g.off.end() - 1);
  int64_t pairs2 = 0;
  lacs::for_each_int_pair(in, [&](int64_t a, int64_t b) {
    ++pairs2;
    if (a == b) return;
    if (a >= ids_n || b >= ids_n) throw std::runtime_error("edge list changed between passes");
    const int64_t u = deg[a], v = deg[b];
    if (cur[u] >= g.off[u + 1] || cur[v] >= g.off[v + 1]) throw std::runtime_error("edge list changed between passes");
    g.adj[static_cast<size_t>(cur[u]++)] = static_cast<int32_t>(v);
    g.adj[static_cast<size_t>(cur[v]++)] = static_cast<int32_t>(u);
  });
  if (pairs2 != pairs) throw std::runtime_error("edge list changed between passes");
  std::vector<uint32_t>().swap(deg);
  // sort and deduplicate every list, compacting in place (cur becomes the new offsets)
  int64_t w = 0;
  for (int64_t v = 0; v < n; ++v) {
    auto b = g.adj.begin() + g.off[v], e = g.adj.begin() + g.off[v + 1];
    std::sort(b, e);
    auto last = std::unique(b, e);
    cur[v] = w;
    for (auto it = b; it != last; ++it) g.adj[static_cast<size_t>(w++)] = *it;
  }
  for (int64_t v = 0; v < n; ++v) g.off[v] = cur[v];
  g.off[n] = w;
  g.adj.resize(static_cast<size_t>(w));
  g.adj.shrink_to_fit();
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  std::string in, out, relabel;
  int64_t n_given = -1;
  bool lowmem = false;
  const char* usage = "usage: lacs_convert --in edges.txt --out graph.bin [--relabel ids.txt] [--n N] [--lowmem]\n";
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--in") in = next();
    else if (k == "--out") out = next();
    else if (k == "--relabel") relabel = next();
    else if (k == "--n") n_given = std::stoll(next());
    else if (k == "--lowmem") lowmem = true;
    else {
      std::cerr << usage;
      return 2;
    }
  }
  if (in.empty() || out.empty()) {
    std::cerr << usage;
    return 2;
  }
  try {
    if (lowmem) {
      lacs::Graph g = convert_lowmem(in, relabel, n_given);
      lacs::write_graph(out, g);
      std::cout << "{\"n\":" << g.n << ",\"m\":" << g.num_edges() << ",\"lowmem\":true}" << std::endl;
      return 0;
    }
    auto pairs = lacs::read_int_pairs(in);
    std::vector<std::pair<int32_t, int32_t>> edges;
    edges.reserve(pairs.size());
    int64_t n = 0;
    if (!relabel.empty()) {
      std::vector<int64_t> ids;
      ids.reserve(pairs.size() * 2);
      for (auto& p : pairs) {
        ids.push_back(p.first);
        ids.push_back(p.second);
      }
      std::sort(ids.begin(), ids.end());
      ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      n = static_cast<int64_t>(ids.size());
      for (auto& p : pairs) {
        int64_t a = std::lower_bound(ids.begin(), ids.end(), p.first) - ids.begin();
        int64_t b = std::lower_bound(ids.begin(), ids.end(), p.second) - ids.begin();
        edges.emplace_back(static_cast<int32_t>(a), static_cast<int32_t>(b));
      }
      FILE* f = std::fopen(relabel.c_str(), "w");
      if (!f) throw std::runtime_error("cannot write " + relabel);
      for (auto x : ids) std::fprintf(f, "%lld\n", static_cast<long long>(x));
      std::fclose(f);
    } else {
      int64_t mx = -1;
      for (auto& p : pairs) {
        if (p.first < 0 || p.second < 0) throw std::runtime_error("negative id; use --relabel");
        mx = std::max({mx, p.first, p.second});
        edges.emplace_back(static_cast<int32_t>(p.first), static_cast<int32_t>(p.second));
      }
      n = (n_given >= 0) ? n_given : mx + 1;
    }
    std::vector<std::pair<int64_t, int64_t>>().swap(pairs);
    lacs::Graph g = lacs::build_csr(n, edges);
    lacs::write_graph(out, g);
    std::cout << "{\"n\":" << g.n << ",\"m\":" << g.num_edges() << "}" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
