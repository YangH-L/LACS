"""Write the LaTeX tables of Section 6 from the result bundles (no number is typed by hand).

Usage: python3 -I make_tables.py <work_dir> <paper_tables_dir>
  <work_dir>/derived/w2_numbers.json      (w2_numbers.py: W2 bundle + round 3)
  <work_dir>/p2x/p2x_summary.json          (P2X: Friendster k-core reference; p2x bundle)
  <work_dir>/derived/p0_e3_numbers.json   (p0_e3_numbers.py: P0 bundle, E3)
  <work_dir>/p0bundle/runs_p0/report/e1_summary.csv          (P0 bundle, E1)
  <work_dir>/r4res/runs/<gowalla|brightkite>/results/r4/geo_e2.json   (round 4)
  <work_dir>/paper_base/notes/analysis/geo_size_check.json, geo_posthoc.json (round 4, post hoc)
  <work_dir>/e5bundle/runs/e5_main/results/e5_analysis.json   (E5, server analysis)
  <work_dir>/e5bundle/runs/e6/results/e6_static.json          (E6 Q2)
Also writes meetup_values.json next to this script (the meetup values, unrounded) for make_fig_meetup.py.
"""
import json
import math
import os
import sys

import pandas as pd

W, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def fmt_ms(x):
    """Milliseconds with 2 significant digits; seconds from 1000 ms on; microseconds below 0.01 ms."""
    if x is None or x != x:
        return "--"
    if x >= 1000:
        s = x / 1000
        return (f"{s:.1f}" if s < 100 else f"{s:.0f}") + "\\,s"
    if x < 0.01:
        us = x * 1000
        return (f"{us:.1f}" if us < 10 else f"{us:.0f}") + "\\,\\textmu{}s"
    if x < 1:
        return f"{x:.2f}"
    if x < 10:
        return f"{x:.1f}"
    return f"{x:.0f}"


def fmt_us(x):
    """Microseconds without a unit, for a column whose header carries the unit (all values below 0.01 ms)."""
    assert x < 0.01, x
    us = x * 1000
    return f"{us:.1f}" if us < 10 else f"{us:.0f}"


def r0(x):
    """Round half up (Python's round() and format() round half to even)."""
    return int(math.floor(x + 0.5 + 1e-9))


def thou(n):
    return f"{n:,}".replace(",", "{,}")


def short(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f}\\,B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}\\,M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}\\,K"
    return str(n)


# ---------------- Table: efficiency at the default setting (W2) ----------------
w2 = json.load(open(f"{W}/derived/w2_numbers.json"))
# W2 did not time the k-core reference on Friendster; P2X (10-27) did, with the W2 engine sources and the same
# 200 queries (p2x_bundle_20261027_132607.tar.gz, sha256 3cd06bc5...f32b): median over the 200 queries, all "ok".
p2x = f"{W}/p2x/p2x_summary.json"
if os.path.exists(p2x) and w2["default"]["friendster"]["kcore_median_ms"] is None:
    fs = json.load(open(p2x))["friendster"]
    assert fs["complete"] and fs["all_ok"], fs
    w2["default"]["friendster"]["kcore_median_ms"] = fs["time_ms"]["median"]
order = [("gowalla", "Gowalla"), ("brightkite", "Brightkite"), ("dblp", "DBLP"), ("youtube", "YouTube"),
         ("pokec", "Pokec"), ("livejournal", "LiveJournal"), ("orkut", "Orkut"), ("friendster", "Friendster")]
lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
         r" & & & & \multicolumn{1}{c}{No LCC} & \multicolumn{3}{c}{LCC exists} & \multicolumn{2}{c}{Greedy} & \\",
         r"\cmidrule(lr){5-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
         r"Graph & $|V|$ & $|E|$ & With LCC & Median (\textmu{}s) & Median & p90 & Unproven & Finds & Size & $k$-core \\",
         r"\midrule"]
for key, name in order:
    r = w2["default"][key]
    lcc_pct = 100 * r["lcc"] / r["n"]
    kc = fmt_ms(r["kcore_median_ms"]) if r["kcore_median_ms"] is not None else "--"
    row = [name, short(r["n_vertices"]), short(r["n_edges"]), f"{r0(lcc_pct)}\\%",
           fmt_us(r["exact_no_lcc_median_ms"]), fmt_ms(r["exact_lcc_median_ms"]), fmt_ms(r["exact_lcc_p90_ms"]),
           str(r["exact_timeouts"]), f"{r0(100 * r['greedy_found_rate_lcc'])}\\%",
           f"{r['greedy_ratio_mean_proven']:.2f}", kc]
    lines.append(" & ".join(row) + r" \\")
    if key == "brightkite":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_eff.tex", "w").write("\n".join(lines) + "\n")

# ---------------- Table: exact solvers in one sweep framework (P0 E3) ----------------
e3 = json.load(open(f"{W}/derived/p0_e3_numbers.json"))
names = [("bnb", r"Branch and bound (ours)"), ("krmax_simple", r"AdvMax, simple branching"),
         ("krmax", r"AdvMax"), ("ilp", r"MILP (HiGHS)"), ("cpsat", r"CP-SAT")]
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Proven} & \multicolumn{2}{c}{Proven, LCC} & \multicolumn{2}{c}{Slowdown} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Solver & LJ & Ork. & LJ & Ork. & LJ & Ork. \\", r"\midrule"]
for key, name in names:
    lj, ok = e3["livejournal"][key], e3["orkut"][key]
    def sl(rec):
        return "1" if key == "bnb" else (f"{rec['ratio_hard_median']:.1f}" if rec['ratio_hard_median'] < 100 else f"{rec['ratio_hard_median']:.0f}")
    lines.append(" & ".join([name, f"{100 * lj['proven']:.1f}", f"{100 * ok['proven']:.1f}",
                             f"{100 * lj['proven_lcc']:.1f}", f"{100 * ok['proven_lcc']:.1f}", sl(lj), sl(ok)]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_solvers.tex", "w").write("\n".join(lines) + "\n")

# ---------------- Table: dynamics on returned groups (P0 E1) ----------------
e1 = pd.read_csv(f"{W}/p0bundle/runs_p0/report/e1_summary.csv")
meths = [("lcc_exact", "LCC (Exact)"), ("pairwise", "Pairwise"), ("kr_tau", r"$(k,r)$-core at $\tau$"),
         ("sac_sized", "Size-matched SAC")]
vars_ = [("hk_eq_G", "Equal"), ("hk_rand_G_a0.1", "Concentrated"), ("dw_G", "Gossip")]
dss = ["gowalla", "brightkite", "livejournal", "orkut"]


def pct_int(x):
    v = 100 * x
    if v == 0:
        return "0"
    if v < 1:
        return "$<$1"
    return f"{v:.0f}"


lines = [r"\begin{tabular}{l" + "rrrr" * 3 + "}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{4}{c}{" + t + "}" for _, t in vars_) + r" \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
         "Method & " + " & ".join(["G", "B", "L", "O"] * 3) + r" \\", r"\midrule"]
for mk, mn in meths:
    cells = []
    for vk, _ in vars_:
        for ds in dss:
            x = e1[(e1.dataset == ds) & (e1.method == mk) & (e1.variant == vk)].share_breakable
            # Python's round() rounds half to even; use explicit half-up to match the P0 results doc
            v = float(x.iloc[0])
            cells.append("0" if v == 0 else ("$<$1" if 100 * v < 1 else str(int(100 * v + 0.5 + 1e-9))))
    lines.append(mn + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_dyn.tex", "w").write("\n".join(lines) + "\n")

# ---------------- Table: meetups, LCC minus size-matched SAC (pp) ----------------
r4 = {ds: json.load(open(f"{W}/r4res/runs/{ds}/results/r4/geo_e2.json")) for ds in ["gowalla", "brightkite"]}
sc = json.load(open(f"{W}/paper_base/notes/analysis/geo_size_check.json"))
ph = json.load(open(f"{W}/paper_base/notes/analysis/geo_posthoc.json"))
e5 = json.load(open(f"{W}/e5bundle/runs/e5_main/results/e5_analysis.json"))


def c(m, lo, hi):
    sign = "+" if m >= 0 else "-"
    return f"${sign}{abs(m):.2f}$\\,{{\\scriptsize[{lo:.2f},{hi:.2f}]}}".replace("[-", "[$-$").replace(",-", ",$-$")


rows = []
# pair rate raw
g, b = r4["gowalla"]["paired_covisit_pair_rate_vs_lcc_exact"]["sac_sized"], r4["brightkite"]["paired_covisit_pair_rate_vs_lcc_exact"]["sac_sized"]
y = e5["H1"]["sac_sized"]["metrics"]["covisit_pair_rate"]
rows.append(("Pairs, raw", c(100 * g["mean_diff"], 100 * g["ci95"][0], 100 * g["ci95"][1]),
             c(100 * b["mean_diff"], 100 * b["ci95"][0], 100 * b["ci95"][1]),
             c(100 * y["raw"]["mean_diff"], 100 * y["raw"]["ci95"][0], 100 * y["raw"]["ci95"][1])))
ga, ba = sc["gowalla"]["sac_sized"]["pair_diff_equal_nonidentical_pp"], sc["brightkite"]["sac_sized"]["pair_diff_equal_nonidentical_pp"]
ya = y["size_a_same_size_different_members"]
rows.append(("\\quad (a) same size", c(ga["mean"], *ga["ci95"]), c(ba["mean"], *ba["ci95"]),
             c(100 * ya["mean_diff"], 100 * ya["ci95"][0], 100 * ya["ci95"][1])))
gb, bb = ph["gowalla"]["h1_size_control_sac_sized"], ph["brightkite"]["h1_size_control_sac_sized"]
yb = y["size_b_regression"]["intercept"]
rows.append(("\\quad (b) adjusted", c(gb["intercept_pp"], *gb["intercept_ci95"]), c(bb["intercept_pp"], *bb["intercept_ci95"]),
             c(100 * yb["coef"], 100 * yb["ci95"][0], 100 * yb["ci95"][1])))
g, b = r4["gowalla"]["paired_member_meet_rate_vs_lcc_exact"]["sac_sized"], r4["brightkite"]["paired_member_meet_rate_vs_lcc_exact"]["sac_sized"]
y = e5["H1"]["sac_sized"]["metrics"]["member_meet_rate"]
rows.append(("Members, raw", c(100 * g["mean_diff"], 100 * g["ci95"][0], 100 * g["ci95"][1]),
             c(100 * b["mean_diff"], 100 * b["ci95"][0], 100 * b["ci95"][1]),
             c(100 * y["raw"]["mean_diff"], 100 * y["raw"]["ci95"][0], 100 * y["raw"]["ci95"][1])))
ga, ba = sc["gowalla"]["sac_sized"]["member_diff_equal_nonidentical_pp"], sc["brightkite"]["sac_sized"]["member_diff_equal_nonidentical_pp"]
ya = y["size_a_same_size_different_members"]
rows.append(("\\quad (a) same size", c(ga["mean"], *ga["ci95"]), c(ba["mean"], *ba["ci95"]),
             c(100 * ya["mean_diff"], 100 * ya["ci95"][0], 100 * ya["ci95"][1])))
nq = (thou(r4["gowalla"]["paired_covisit_pair_rate_vs_lcc_exact"]["sac_sized"]["n"]) + " / " + thou(sc["gowalla"]["sac_sized"]["pair_diff_equal_nonidentical_pp"]["n"]),
      thou(r4["brightkite"]["paired_covisit_pair_rate_vs_lcc_exact"]["sac_sized"]["n"]) + " / " + thou(sc["brightkite"]["sac_sized"]["pair_diff_equal_nonidentical_pp"]["n"]),
      thou(e5["H1"]["sac_sized"]["n_common"]) + " / " + thou(e5["H1"]["sac_sized"]["n_same_size_different_members"]))
# The same values, unrounded, for the figure version of this table (make_fig_meetup.py).
def _v(m, lo, hi):
    return {"est": m, "lo": lo, "hi": hi}


_g, _b = r4["gowalla"], r4["brightkite"]
_y = e5["H1"]["sac_sized"]["metrics"]
meetup_vals = {"unit": "percentage points (LCC minus size-matched SAC)", "datasets": {}}
for _ds, _r4, _sc, _ph in [("Gowalla", _g, sc["gowalla"], ph["gowalla"]), ("Brightkite", _b, sc["brightkite"], ph["brightkite"])]:
    _pr = _r4["paired_covisit_pair_rate_vs_lcc_exact"]["sac_sized"]
    _mr = _r4["paired_member_meet_rate_vs_lcc_exact"]["sac_sized"]
    _pa = _sc["sac_sized"]["pair_diff_equal_nonidentical_pp"]
    _ma = _sc["sac_sized"]["member_diff_equal_nonidentical_pp"]
    _pb = _ph["h1_size_control_sac_sized"]
    meetup_vals["datasets"][_ds] = {
        "pairs": {"raw": _v(100 * _pr["mean_diff"], 100 * _pr["ci95"][0], 100 * _pr["ci95"][1]),
                  "a": _v(_pa["mean"], *_pa["ci95"]), "b": _v(_pb["intercept_pp"], *_pb["intercept_ci95"])},
        "members": {"raw": _v(100 * _mr["mean_diff"], 100 * _mr["ci95"][0], 100 * _mr["ci95"][1]),
                    "a": _v(_ma["mean"], *_ma["ci95"])},
        "n_raw": _pr["n"], "n_a": _pa["n"]}
_yp, _ym = _y["covisit_pair_rate"], _y["member_meet_rate"]
meetup_vals["datasets"]["Yelp"] = {
    "pairs": {"raw": _v(100 * _yp["raw"]["mean_diff"], 100 * _yp["raw"]["ci95"][0], 100 * _yp["raw"]["ci95"][1]),
              "a": _v(100 * _yp["size_a_same_size_different_members"]["mean_diff"], *[100 * t for t in _yp["size_a_same_size_different_members"]["ci95"]]),
              "b": _v(100 * _yp["size_b_regression"]["intercept"]["coef"], *[100 * t for t in _yp["size_b_regression"]["intercept"]["ci95"]])},
    "members": {"raw": _v(100 * _ym["raw"]["mean_diff"], 100 * _ym["raw"]["ci95"][0], 100 * _ym["raw"]["ci95"][1]),
                "a": _v(100 * _ym["size_a_same_size_different_members"]["mean_diff"], *[100 * t for t in _ym["size_a_same_size_different_members"]["ci95"]])},
    "n_raw": e5["H1"]["sac_sized"]["n_common"], "n_a": e5["H1"]["sac_sized"]["n_same_size_different_members"]}
json.dump(meetup_vals, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "meetup_values.json"), "w"), indent=1)

lines = [r"\begin{tabular}{lrrr}", r"\toprule", r" & Gowalla & Brightkite & Yelp \\", r"\midrule"]
for r_ in rows:
    lines.append(" & ".join(r_) + r" \\")
lines.append(r"\midrule")
lines.append("Queries, all / (a) & " + " & ".join(nq) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_meetup.tex", "w").write("\n".join(lines) + "\n")

# ---------------- Table: Congress Q2 (static tolerance violations, all queries) ----------------
q2 = pd.read_csv(f"{W}/e5bundle/runs/e6/results/e6_static.csv")
q2 = q2[q2.scope == "all"].set_index("method")
def p1(x):
    v = 100 * x
    return "0" if v == 0 else ("100" if v == 100 else (f"{v:.1f}"))
rowsq = [("LCC (Exact, Greedy, BallCore-F)", ["lcc_exact", "lcc_greedy", "lcc_ballcore_f"]),
         ("Pairwise", ["pairwise"]), (r"$(k,r)$-core at $\tau$", ["kr_tau"]), ("Size-matched SAC", ["sac_sized"]),
         ("SAC", ["sac"]), (r"Global $(k,r)$-core", ["kr_global"]), ("BallCore-R", ["lcc_ballcore_r"]),
         (r"Random, $k$-core, edge-accept", ["random_bfs", "kcore", "edgeacc"])]
lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Method & (iii) holds & Members & Pairs \\", r"\midrule"]
for name, ms in rowsq:
    vals = []
    for col in ["share_iii", "share_members_ecc_gt_theta", "share_pairs_rejected"]:
        xs = [float(q2.loc[m, col]) for m in ms]
        lo, hi = min(xs), max(xs)
        vals.append(p1(lo) if p1(lo) == p1(hi) else f"{p1(lo)}--{p1(hi)}")
    lines.append(name + " & " + " & ".join(vals) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{OUT}/tab_congress.tex", "w").write("\n".join(lines) + "\n")
print("tables written to", OUT)
