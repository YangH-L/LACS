// Fast reader for whitespace/comma separated integer pairs (one edge per line).
// Lines starting with '#' or '%' are skipped; extra columns are ignored.
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lacs {

// Streams the pairs to f(a, b) without storing them (v0.5; used by the two-pass converter).
template <class F>
inline void for_each_int_pair(const std::string& path, F f) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open edge list: " + path);
  std::vector<char> buf(1 << 24);
  std::string carry;
  int64_t vals[2];
  int nv = 0;
  bool in_num = false, neg = false, skip_line = false, line_start = true;
  int64_t cur = 0;
  auto flush_num = [&]() {
    if (in_num) {
      if (nv < 2) vals[nv] = neg ? -cur : cur;
      ++nv;
    }
    in_num = false;
    neg = false;
    cur = 0;
  };
  auto end_line = [&]() {
    flush_num();
    if (!skip_line && nv >= 2) f(vals[0], vals[1]);
    nv = 0;
    skip_line = false;
    line_start = true;
  };
  size_t got;
  while ((got = std::fread(buf.data(), 1, buf.size(), fp)) > 0) {
    for (size_t i = 0; i < got; ++i) {
      char c = buf[i];
      if (c == '\n') {
        end_line();
        continue;
      }
      if (skip_line) continue;
      if (line_start && (c == '#' || c == '%')) {
        skip_line = true;
        continue;
      }
      line_start = false;
      if (c >= '0' && c <= '9') {
        in_num = true;
        cur = cur * 10 + (c - '0');
      } else if (c == '-' && !in_num) {
        neg = true;
      } else {
        flush_num();
        if (nv >= 2) skip_line = true;  // ignore remaining columns
      }
    }
  }
  end_line();
  std::fclose(fp);
}

inline std::vector<std::pair<int64_t, int64_t>> read_int_pairs(const std::string& path) {
  std::vector<std::pair<int64_t, int64_t>> out;
  for_each_int_pair(path, [&](int64_t a, int64_t b) { out.emplace_back(a, b); });
  return out;
}

}  // namespace lacs
