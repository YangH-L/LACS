// lacs_project: meta-path projection of a bipartite graph (e.g. user-item,
// user-thread) into a user-user graph. Two users are connected when they share
// at least --min-shared right-side objects (meta-path U-X-U).
//
//   lacs_project --in bipartite.txt --out graph.bin --ids users.txt
//                [--max-right-degree D] [--min-shared c] [--max-pairs P]
//
// Right objects with more than D users are skipped (hub cap; 0 = no cap).
// Left ids are relabelled to 0..n-1 in sorted order and written to users.txt.
#include <algorithm>
#include <iostream>

#include "edge_reader.hpp"
#include "lacs_core.hpp"

int main(int argc, char** argv) {
  std::string in, out, ids_path;
  int64_t max_rdeg = 0, min_shared = 1, max_pairs = 2000000000LL;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--in") in = next();
    else if (k == "--out") out = next();
    else if (k == "--ids") ids_path = next();
    else if (k == "--max-right-degree") max_rdeg = std::stoll(next());
    else if (k == "--min-shared") min_shared = std::stoll(next());
    else if (k == "--max-pairs") max_pairs = std::stoll(next());
    else {
      std::cerr << "usage: lacs_project --in bipartite.txt --out graph.bin --ids users.txt "
                   "[--max-right-degree D] [--min-shared c] [--max-pairs P]\n";
      return 2;
    }
  }
  if (in.empty() || out.empty() || ids_path.empty()) {
    std::cerr << "usage: lacs_project --in bipartite.txt --out graph.bin --ids users.txt ...\n";
    return 2;
  }
  try {
    auto pairs = lacs::read_int_pairs(in);
    std::vector<int64_t> lids, rids;
    for (auto& p : pairs) {
      lids.push_back(p.first);
      rids.push_back(p.second);
    }
    std::sort(lids.begin(), lids.end());
    lids.erase(std::unique(lids.begin(), lids.end()), lids.end());
    std::sort(rids.begin(), rids.end());
    rids.erase(std::unique(rids.begin(), rids.end()), rids.end());
    // (right, left) pairs, deduplicated
    std::vector<std::pair<int64_t, int32_t>> rl;
    rl.reserve(pairs.size());
    for (auto& p : pairs) {
      int64_t l = std::lower_bound(lids.begin(), lids.end(), p.first) - lids.begin();
      int64_t r = std::lower_bound(rids.begin(), rids.end(), p.second) - rids.begin();
      rl.emplace_back(r, static_cast<int32_t>(l));
    }
    std::vector<std::pair<int64_t, int64_t>>().swap(pairs);
    std::sort(rl.begin(), rl.end());
    rl.erase(std::unique(rl.begin(), rl.end()), rl.end());
    std::vector<uint64_t> keys;
    int64_t skipped = 0;
    for (size_t a = 0; a < rl.size();) {
      size_t b = a;
      while (b < rl.size() && rl[b].first == rl[a].first) ++b;
      int64_t d = static_cast<int64_t>(b - a);
      if (max_rdeg > 0 && d > max_rdeg) {
        ++skipped;
      } else {
        for (size_t i = a; i < b; ++i)
          for (size_t j = i + 1; j < b; ++j) {
            uint32_t u = static_cast<uint32_t>(rl[i].second), v = static_cast<uint32_t>(rl[j].second);
            if (u > v) std::swap(u, v);
            keys.push_back((static_cast<uint64_t>(u) << 32) | v);
          }
        if (static_cast<int64_t>(keys.size()) > max_pairs)
          throw std::runtime_error("projection exceeds --max-pairs; lower --max-right-degree");
      }
      a = b;
    }
    std::sort(keys.begin(), keys.end());
    std::vector<std::pair<int32_t, int32_t>> edges;
    for (size_t a = 0; a < keys.size();) {
      size_t b = a;
      while (b < keys.size() && keys[b] == keys[a]) ++b;
      if (static_cast<int64_t>(b - a) >= min_shared)
        edges.emplace_back(static_cast<int32_t>(keys[a] >> 32), static_cast<int32_t>(keys[a] & 0xffffffffULL));
      a = b;
    }
    std::vector<uint64_t>().swap(keys);
    lacs::Graph g = lacs::build_csr(static_cast<int64_t>(lids.size()), edges);
    lacs::write_graph(out, g);
    FILE* f = std::fopen(ids_path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + ids_path);
    for (auto x : lids) std::fprintf(f, "%lld\n", static_cast<long long>(x));
    std::fclose(f);
    std::cout << "{\"n\":" << g.n << ",\"m\":" << g.num_edges() << ",\"right_objects\":" << rids.size()
              << ",\"skipped_hubs\":" << skipped << "}" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
