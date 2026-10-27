// lacs_search: run community search queries and print one JSON object per
// (query, algorithm) line.
//
//   lacs_search --graph G.bin --pos P.bin --theta T.bin --queries Q.txt
//               --algo exact,ballcore_r,kcore --k 3 [options] [--out res.jsonl]
//   lacs_search --graph G.bin --info        (prints n, m and the core histogram)
//
// Positions and theta are LACSMT01 matrices (theta: n x 1). Queries: one vertex
// id per line, optionally followed by a per-query threshold r for the kr_*
// algorithms (overrides --r; '-' = none) and a per-query minimum size for sac_inc.
// Search runs in a thread with a large stack because the
// branch-and-bound recursion depth can reach the candidate-set size.
#include <pthread.h>

#include <iostream>

#include "lacs_core.hpp"

namespace {

struct Args {
  std::string graph, pos, theta, queries, out, cores_out;
  std::vector<std::string> algos{"exact"};
  lacs::Options opt;
  bool info = false;
  bool members = true;
};

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(item);
  return out;
}

void usage() {
  std::cerr << "usage: lacs_search --graph G.bin [--info [--cores-out C.bin]] | --pos P.bin --theta T.bin --queries Q.txt\n"
               "       [--algo a,b,...] [--k 3] [--time-limit 600] [--max-exact-size 3000]\n"
               "       [--max-greedy-size 20000] [--greedy-max-tau 20] [--r R] [--diam-cap 5000]\n"
               "       [--no-ub-order] [--no-color-bound] [--no-seed] [--no-compat-bound] [--no-edge-filter]\n"
               "       [--no-local] [--no-index]\n"
               "       [--tau-order bound|count] [--color-order degree|index] [--no-ub] [--ub-time S]\n"
               "       [--max-bound-size 20000]\n"
               "       [--no-members] [--out file]\n"
               "algos: kcore ballcore_f ballcore_r greedy exact kr_exact kr_greedy\n"
               "       pairwise_exact pairwise_greedy edgeacc sac_inc\n"
               "queries file: one vertex id per line, optionally 'q r [s]' (per-query r for kr_*, '-' = none;\n"
               "              s = minimum community size for sac_inc)\n";
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
      return argv[++i];
    };
    if (k == "--graph") a.graph = next();
    else if (k == "--pos") a.pos = next();
    else if (k == "--theta") a.theta = next();
    else if (k == "--queries") a.queries = next();
    else if (k == "--out") a.out = next();
    else if (k == "--algo") a.algos = split_csv(next());
    else if (k == "--k") a.opt.k = std::stoi(next());
    else if (k == "--time-limit") a.opt.time_limit_s = std::stod(next());
    else if (k == "--max-exact-size") a.opt.max_exact_size = std::stoll(next());
    else if (k == "--max-greedy-size") a.opt.max_greedy_size = std::stoll(next());
    else if (k == "--greedy-max-tau") a.opt.greedy_max_tau = std::stoi(next());
    else if (k == "--r") a.opt.r = std::stod(next());
    else if (k == "--diam-cap") a.opt.diam_cap = std::stoll(next());
    else if (k == "--no-ub-order") a.opt.use_ub_order = false;
    else if (k == "--no-color-bound") a.opt.use_color_bound = false;
    else if (k == "--no-seed") a.opt.use_seed = false;
    else if (k == "--no-compat-bound") a.opt.use_compat_bound = false;
    else if (k == "--no-edge-filter") a.opt.use_edge_filter = false;
    else if (k == "--no-local") a.opt.use_local = false;
    else if (k == "--no-index") a.opt.use_index = false;
    else if (k == "--color-order") {
      const std::string v = next();
      if (v == "degree") a.opt.color_order_degree = true;
      else if (v == "index") a.opt.color_order_degree = false;
      else throw std::runtime_error("--color-order must be degree or index");
    }
    else if (k == "--tau-order") {
      const std::string v = next();
      if (v == "bound") a.opt.tau_order_bound = true;
      else if (v == "count") a.opt.tau_order_bound = false;
      else throw std::runtime_error("--tau-order must be bound or count");
    }
    else if (k == "--no-ub") a.opt.compute_ub = false;
    else if (k == "--ub-time") a.opt.ub_time_s = std::stod(next());
    else if (k == "--max-bound-size") a.opt.max_bound_size = std::stoll(next());
    else if (k == "--no-members") a.members = false;
    else if (k == "--info") a.info = true;
    else if (k == "--cores-out") a.cores_out = next();
    else if (k == "-h" || k == "--help") {
      usage();
      std::exit(0);
    } else
      throw std::runtime_error("unknown argument: " + k);
  }
  if (a.graph.empty()) throw std::runtime_error("--graph is required");
  if (!a.info && (a.pos.empty() || a.theta.empty() || a.queries.empty()))
    throw std::runtime_error("--pos, --theta and --queries are required");
  return a;
}

struct Job {
  Args args;
  int rc = 0;
  std::string err;
};

void* run_job(void* p) {
  Job* job = static_cast<Job*>(p);
  try {
    const Args& a = job->args;
    lacs::Graph g = lacs::read_graph(a.graph);
    if (a.info) {
      auto core = lacs::core_numbers(g);
      int mc = 0;
      for (auto c : core) mc = std::max(mc, c);
      std::vector<int64_t> hist(static_cast<size_t>(mc) + 1, 0);
      for (auto c : core) hist[c]++;
      std::cout << "{\"n\":" << g.n << ",\"m\":" << g.num_edges() << ",\"max_core\":" << mc << ",\"core_ge\":[";
      int64_t acc = 0;
      std::vector<int64_t> ge(hist.size());
      for (int c = mc; c >= 0; --c) {
        acc += hist[c];
        ge[c] = acc;
      }
      for (size_t c = 0; c < ge.size(); ++c) std::cout << (c ? "," : "") << ge[c];
      std::cout << "]}" << std::endl;
      if (!a.cores_out.empty()) {  // raw int32 core number per vertex
        std::ofstream co(a.cores_out, std::ios::binary);
        if (!co) throw std::runtime_error("cannot write " + a.cores_out);
        co.write(reinterpret_cast<const char*>(core.data()), static_cast<std::streamsize>(sizeof(int32_t) * core.size()));
      }
      return nullptr;
    }
    lacs::Matrix pos = lacs::read_matrix(a.pos);
    lacs::Matrix th = lacs::read_matrix(a.theta);
    if (th.rows != g.n || th.cols != 1) throw std::runtime_error("theta must be an n x 1 matrix");
    std::vector<float> theta(th.data.begin(), th.data.end());
    struct Query {
      int64_t q;
      double r;      // per-query r (NaN: use --r)
      int64_t size;  // per-query minimum size for sac_inc (-1: none)
    };
    std::vector<Query> queries;
    {
      std::ifstream in(a.queries);
      if (!in) throw std::runtime_error("cannot open queries: " + a.queries);
      std::string line;
      while (std::getline(in, line)) {
        std::istringstream ls(line);
        int64_t q;
        if (!(ls >> q)) continue;
        Query x{q, std::numeric_limits<double>::quiet_NaN(), -1};
        std::string t;
        if (ls >> t && t != "-") x.r = std::stod(t);
        if (ls >> t && t != "-") x.size = std::stoll(t);
        queries.push_back(x);
      }
    }
    lacs::Searcher s(g, pos, theta);
    std::ofstream fout;
    std::ostream* out = &std::cout;
    if (!a.out.empty()) {
      fout.open(a.out);
      if (!fout) throw std::runtime_error("cannot write: " + a.out);
      out = &fout;
    }
    for (const auto& qr : queries) {
      lacs::Options o = a.opt;
      if (std::isfinite(qr.r)) o.r = qr.r;
      if (qr.size >= 0) o.min_size = qr.size;
      for (auto& algo : a.algos) {
        lacs::Result r = s.run(algo, qr.q, o);
        (*out) << lacs::to_json(r, a.members) << "\n";
      }
      out->flush();
    }
  } catch (const std::exception& e) {
    job->rc = 1;
    job->err = e.what();
  }
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  Job job;
  try {
    job.args = parse(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    usage();
    return 2;
  }
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, static_cast<size_t>(1) << 30);  // 1 GiB stack
  pthread_t th;
  if (pthread_create(&th, &attr, run_job, &job) != 0) {
    run_job(&job);  // fall back to the main thread
  } else {
    pthread_join(th, nullptr);
  }
  pthread_attr_destroy(&attr);
  if (job.rc != 0) std::cerr << "error: " << job.err << "\n";
  return job.rc;
}
