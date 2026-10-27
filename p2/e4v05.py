"""P2-5: P0's E4 on the round-3 instances with the v0.5 engine (docs/PLANS.md (lacs_p2_plan_20261026) section 5).

  PYTHONPATH=p0:. python -m p2.e4v05 run --config p2/configs/e4v05.yaml --dataset livejournal --experiment e4_scale \
      --p0-root runs_p0
  PYTHONPATH=p0:. python -m p2.e4v05 summary --root runs_p2/e4v05 --p0-root runs_p0 \
      --out runs_p2/e4v05/report

run: P0's own driver (p0kit.engine_sweeps.run_setting, unchanged) with this project's engine. Run from the
v0.5 tree with PYTHONPATH=p0:., `lacs.search.cpp` is the v0.5 wrapper and cpp/build/lacs_search the v0.5
engine, so nothing in P0's code changes. Added around it:
  * the replay of the round-3 random stream must be exact (else exit code 3, nothing run);
  * engine processes run under the stall watchdog of p2.watchdog;
  * P0's per-setting instance export for E1 (bridge.save_instance: graph, positions and tolerances as .npy,
    several GB) is skipped, it does not change any result and E1 is not part of P2-5;
  * after each setting, the gate of plan section 5.2: every query proven optimal by both v0.4 (P0's
    per_query.csv) and v0.5 has the same optimum, else exit code 4 and no further setting is run.
Settings with DONE are skipped on a rerun (P0's behaviour); a finished setting is gated again.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "p0", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LCC_WITNESS = ("exact", "greedy", "ballcore_f")  # methods whose non-empty answers satisfy (iii)


def _engine_info() -> dict:
    import hashlib

    from lacs.search import cpp as cppmod

    exe = cppmod.BUILD_DIR / "lacs_search"
    stamp = cppmod.BUILD_DIR / ".sources.sha256"
    return {"engine": str(exe), "engine_sha256": hashlib.sha256(exe.read_bytes()).hexdigest() if exe.exists() else None,
            "engine_sources_sha256": stamp.read_text().strip() if stamp.exists() else None,
            "sources_digest_now": cppmod._sources_digest()}


def gate_setting(mine_dir: Path, p0_dir: Path) -> dict:
    a = pd.read_csv(mine_dir / "per_query.csv")
    b = pd.read_csv(p0_dir / "per_query.csv")
    a = a[a["algo"] == "exact"].set_index("q")
    b = b[b["algo"] == "exact"].set_index("q")
    common = a.index.intersection(b.index)
    both = [q for q in common if a.at[q, "status"] == "optimal" and b.at[q, "status"] == "optimal"]
    mism = [q for q in both if int(a.at[q, "size"]) != int(b.at[q, "size"])]
    res = {"setting": mine_dir.name, "queries_v05": int(len(a)), "queries_v04": int(len(b)), "common": int(len(common)),
           "same_query_set": bool(set(a.index) == set(b.index)), "both_proven": len(both), "mismatches": len(mism),
           "examples": [{"q": int(q), "v05": int(a.at[q, "size"]), "v04": int(b.at[q, "size"])} for q in mism[:20]],
           "proven_v05_only": int(sum(1 for q in common if a.at[q, "status"] == "optimal" and b.at[q, "status"] != "optimal")),
           "proven_v04_only": int(sum(1 for q in common if a.at[q, "status"] != "optimal" and b.at[q, "status"] == "optimal"))}
    res["pass"] = bool(res["same_query_set"] and len(mism) == 0)
    return res


def cmd_run(args) -> int:
    from lacs.utils import setup_logging

    from p0kit import bridge, engine_sweeps
    from p0kit.runner_common import load_yaml, write_json

    from p2 import watchdog

    setup_logging("WARNING")
    cfg = load_yaml(args.config)
    if args.e3_run:
        cfg["e3_run"] = args.e3_run
    if args.out_root:
        cfg["out_root"] = args.out_root
    e3_run = cfg["e3_run"]
    watchdog.install()
    saved = bridge.save_instance
    bridge.save_instance = lambda *a, **k: None  # P2-5 needs no E1 instances (see the module docstring)
    try:
        return _run(args, cfg, e3_run, bridge, engine_sweeps, write_json)
    finally:
        bridge.save_instance = saved


def _run(args, cfg, e3_run, bridge, engine_sweeps, write_json) -> int:
    ds = bridge.load_round3(e3_run, args.dataset)
    rep = bridge.replay_synthetic(ds)
    print(f"[{args.dataset}] replay of the round-3 random stream: exact={rep['exact']} "
          f"(positions {rep.get('check_positions')}, tolerances {rep.get('check_theta')})", flush=True)
    if not rep["exact"]:
        print("!! the round-3 random stream does not replay exactly: stopping (plan section 5.1)")
        return 3
    r_nbr = bridge.median_edge_distance(ds["indptr"], ds["indices"], ds["X0"], seed=int(cfg.get("seed", 0)))
    qinfo = bridge.default_queries(e3_run, args.dataset, k=ds["k0"], dim=ds["dim"])
    n_proc = int(args.n_proc or cfg.get("n_proc", 1))
    info = _engine_info()
    print(f"[{args.dataset}] engine {info['engine']} sha256 {info['engine_sha256']}; built from sources "
          f"{info['engine_sources_sha256']} (now {info['sources_digest_now']})", flush=True)
    if info["engine_sources_sha256"] != info["sources_digest_now"]:
        print("!! the engine binary was not built from the current sources (run the build stage)")
        return 2
    rc = 0
    for st in engine_sweeps._settings(cfg, args.experiment):
        if args.only and st["name"] not in args.only.split(","):
            continue
        out = Path(engine_sweeps.run_setting(cfg, args.dataset, args.experiment, st, ds, rep, r_nbr, qinfo, n_proc,
                                             args.force))
        man = json.loads((out / "manifest.json").read_text())
        if "engine" not in man:
            man.update({"engine": info, "p2": "P2-5 v0.5 rerun of P0's E4"})
            write_json(str(out / "manifest.json"), man)
        if args.p0_root:
            p0_dir = Path(args.p0_root) / args.dataset / args.experiment / st["name"]
            if not (p0_dir / "per_query.csv").exists():
                print(f"!! no P0 reference {p0_dir / 'per_query.csv'}: cannot gate")
                return 4
            g = gate_setting(out, p0_dir)
            g["p0_dir"] = str(p0_dir)
            write_json(str(out / "gate_v04.json"), g)
            print(f"[gate] {args.dataset} {args.experiment} {st['name']}: {g['both_proven']} proven by both, "
                  f"{g['mismatches']} mismatches, same queries {g['same_query_set']}, "
                  f"newly proven {g['proven_v05_only']}, no longer proven {g['proven_v04_only']} -> "
                  f"{'PASS' if g['pass'] else 'FAIL'}", flush=True)
            if not g["pass"]:
                print("!! gate failed: stopping P2-5 (plan section 5.2)")
                return 4
    return rc


def _one(df: pd.DataFrame, algo: str) -> pd.DataFrame:
    return df[df["algo"] == algo].set_index("q")


def summarize(root: Path, p0_root: Path) -> pd.DataFrame:
    rows = []
    for pq in sorted(root.glob("*/e4_*/*/per_query.csv")):
        ds, exp, st = pq.parts[-4], pq.parts[-3], pq.parts[-2]
        new = pd.read_csv(pq)
        p0f = p0_root / ds / exp / st / "per_query.csv"
        old = pd.read_csv(p0f) if p0f.exists() else None
        x5 = _one(new, "exact")
        qs = x5.index
        wit = pd.Series(False, index=qs)
        for algo in LCC_WITNESS:
            for d in (new, old):
                if d is not None and algo in set(d["algo"]):
                    s = _one(d, algo)["size"].reindex(qs).fillna(0)
                    wit |= s >= 2
        has = wit[wit].index
        r = {"dataset": ds, "experiment": exp, "setting": st, "queries": int(len(qs)),
             "has_lcc_share": float(wit.mean()), "has_lcc": int(len(has))}
        for tag, d in (("v05", new), ("v04", old)):
            if d is None:
                continue
            x = _one(d, "exact").reindex(has)
            gr = _one(d, "greedy").reindex(has) if "greedy" in set(d["algo"]) else None
            r[f"{tag}_proven_share_lcc"] = float((x["status"] == "optimal").mean()) if len(has) else np.nan
            r[f"{tag}_timeout_share_lcc"] = float((x["status"] == "timeout").mean()) if len(has) else np.nan
            r[f"{tag}_too_large_share_lcc"] = float((x["status"] == "too_large").mean()) if len(has) else np.nan
            r[f"{tag}_time_median_ms_lcc"] = float(x["time_ms"].median()) if len(has) else np.nan
            if gr is not None and len(has):
                r[f"{tag}_greedy_finds_lcc"] = float((gr["size"] >= 2).mean())
            if tag == "v05" and "ub" in x.columns:
                # plan 5.3: solution / upper bound on the timed-out queries; too_large reported separately
                for lab, st in (("timeout", ["timeout"]), ("too_large", ["too_large"])):
                    un = x[x["status"].isin(st) & (x["ub"] > 0)]
                    ratio = (un["size"] / un["ub"]).astype(float)
                    r[f"v05_{lab}_with_ub"] = int(len(un))
                    r[f"v05_{lab}_size_over_ub_median"] = float(ratio.median()) if len(un) else np.nan
                    r[f"v05_{lab}_size_over_ub_min"] = float(ratio.min()) if len(un) else np.nan
        if old is not None:
            a, b = _one(new, "exact"), _one(old, "exact")
            c = a.index.intersection(b.index)
            r["newly_proven"] = int(((a.loc[c, "status"] == "optimal") & (b.loc[c, "status"] != "optimal")).sum())
            r["no_longer_proven"] = int(((a.loc[c, "status"] != "optimal") & (b.loc[c, "status"] == "optimal")).sum())
            both = c[(a.loc[c, "status"] == "optimal") & (b.loc[c, "status"] == "optimal")]
            r["both_proven"] = int(len(both))
            r["optimum_mismatches"] = int((a.loc[both, "size"] != b.loc[both, "size"]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_summary(args) -> int:
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = summarize(root, Path(args.p0_root))
    if df.empty:
        print("!! no P2-5 results")
        return 1
    order = {"e4_scale": 0, "e4_queries": 1, "e4_mix": 2}
    df = df.sort_values(["experiment", "dataset", "setting"], key=lambda s: s.map(order) if s.name == "experiment" else s)
    df.to_csv(out / "e4v05_summary.csv", index=False)
    cols = [c for c in ["dataset", "experiment", "setting", "queries", "has_lcc_share", "v04_proven_share_lcc",
                        "v05_proven_share_lcc", "v04_timeout_share_lcc", "v05_timeout_share_lcc",
                        "v04_time_median_ms_lcc", "v05_time_median_ms_lcc", "v05_timeout_size_over_ub_median",
                        "v05_timeout_size_over_ub_min", "v04_greedy_finds_lcc", "v05_greedy_finds_lcc", "newly_proven",
                        "no_longer_proven", "optimum_mismatches"] if c in df.columns]
    lines = ["# P2-5: E4 with the v0.5 engine (v0.4 = P0)", "", "| " + " | ".join(cols) + " |",
             "|" + " --- |" * len(cols)]
    for r in df[cols].itertuples(index=False):
        lines.append("| " + " | ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in r) + " |")
    mm = int(df.get("optimum_mismatches", pd.Series([0])).fillna(0).sum())
    lines += ["", f"optimum mismatches over all settings: {mm} (must be 0)"]
    (out / "E4V05_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if mm == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m p2.e4v05", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--experiment", required=True, choices=["e4_scale", "e4_mix", "e4_queries"])
    r.add_argument("--p0-root", default=None, help="P0's runs_p0 (v0.4 reference for the gate)")
    r.add_argument("--e3-run", default=None, help="override the config's e3_run (rehearsals)")
    r.add_argument("--out-root", default=None, help="override the config's out_root (rehearsals)")
    r.add_argument("--only", default=None)
    r.add_argument("--n-proc", type=int, default=None)
    r.add_argument("--force", action="store_true")
    s = sub.add_parser("summary")
    s.add_argument("--root", required=True)
    s.add_argument("--p0-root", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"run": cmd_run, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
