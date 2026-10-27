"""E11 (P2-4): the LCC under scaled tolerances, s * theta_q50 (docs/PLANS.md (lacs_p2_plan_20261026) section 4).

  PYTHONPATH=p0:. python -m p2.e11 prepare-inputs --src runs/gowalla --dst runs_p2/e11/gowalla
  PYTHONPATH=p0:. python -m p2.e11 run   --config p2/configs/e11_gowalla.yaml --s 1 [--mode sample --n 2000]
  PYTHONPATH=p0:. python -m p2.e11 run   --config ... --s 3 --subset 200 --results-name e11_pilot     (pilot)
  PYTHONPATH=p0:. python -m p2.e11 decide --pilot gowalla=DIR brightkite=DIR yelp=DIR --out FILE
  PYTHONPATH=p0:. python -m p2.e11 gate  --run DIR/e11_per_query.csv --ref CSV --ref-kind r4main|e5sweep --out FILE
  PYTHONPATH=p0:. python -m p2.e11 summary --root runs_p2/e11 --out runs_p2/e11/report
  PYTHONPATH=p0:. python -m p2.e11 export-groups --config ... --s 0.8 [query options] --out DIR       (P2-4b)

run: one (dataset, s). The tolerances of users.csv (theta_q50, already floored at 1 km) are multiplied by s
through geo.theta_mult, so the engine (LCC, edgeacc, pairwise acceptance) and the metrics see s * theta. The
queries are geo-e2's (geo._select_queries with the run's seed: mode 'all' = every eligible query, 'sample' =
the same draw as the earlier rounds' parameter studies). Engine passes: exact (LCC, members kept, time limit
60 s) and edgeacc + pairwise_exact (sizes only, 10 s), each in resumable blocks (geo._run_blocks) under a stall
watchdog (p2.watchdog). The LCC groups get round 4's / E5's test-period metrics (geo._member_metrics: co-visit
pair rate, member meet rate, ...; for Yelp these are the co-review metrics). Output, per (dataset, s), in
<out_dir>/results/<results-name>/s<s>/: e11_per_query.csv (internal query ids; no members), e11_summary.json,
config_resolved.json, DONE. Nothing else is written outside <out_dir>/work.

Resuming: a rerun reuses the finished engine blocks cached under <out_dir>/work/<results-name>_s<s>/blocks.
The pilot passes --fresh: its timing decides plan 4.1, so cached blocks of an interrupted pilot are deleted
first and the summary records that no block came from the cache (`decide` refuses otherwise). A run whose
DONE exists is skipped only when its query options (mode, n, subset, limit) equal the requested ones.

export-groups (P2-4b): the LCC groups of a finished E11 run (same config, s and query options), read back
from that run's cached exact blocks without starting the engine, written as P0's group export
(groups.jsonl.gz, per_query.csv, instance.json with the unscaled theta_q50, manifest.json last).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "p0", ROOT):
    if str(_p) not in sys.path:  # also under python -I
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOG = logging.getLogger("p2.e11")
S_GRID = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
UNPROVEN = ("timeout", "too_large")
METRICS = ["diam_km", "min_theta_km", "active_members", "covisit_pair_rate", "member_meet_rate",
           "friend_pair_covisit_rate", "n_covisit_pairs", "members_hash"]
REQUIRED_INPUTS = ["geo/users.csv", "geo/positions.npy", "geo/covisits.npz", "geo/meta.json",
                   "geo/test_checkins.npz", "graph/graph.bin"]


def s_label(s: float) -> str:
    return f"s{float(s):g}"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _dump(obj, path: Path) -> None:
    from lacs.utils import dump_json

    tmp = path.with_suffix(path.suffix + ".tmp")
    dump_json(obj, tmp)
    tmp.replace(path)


# ------------------------------------------------------------------ inputs --
def cmd_prepare_inputs(args) -> int:
    """Copy geo/ and graph/ of a finished run (read-only source) into a new run root, checking sha256."""
    src, dst = Path(args.src), Path(args.dst)
    missing = [f for f in REQUIRED_INPUTS if not (src / f).is_file()]
    if missing:
        print(f"!! {src}: missing {missing}")
        return 2
    rec = {"source": str(src.resolve()), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": {}}
    for sub in ("geo", "graph"):
        for f in sorted((src / sub).iterdir()):
            if not f.is_file():
                continue
            d = dst / sub / f.name
            d.parent.mkdir(parents=True, exist_ok=True)
            h = sha256_file(f)
            if d.exists():
                if sha256_file(d) != h:
                    print(f"!! {d} exists and differs from {f}: remove it by hand if the source changed")
                    return 2
            else:
                tmp = d.with_name(d.name + ".tmp")
                shutil.copyfile(f, tmp)
                if sha256_file(tmp) != h:
                    tmp.unlink()
                    print(f"!! copy of {f} is corrupt")
                    return 2
                tmp.replace(d)
            rec["files"][f"{sub}/{f.name}"] = {"bytes": f.stat().st_size, "sha256": h}
    _dump(rec, dst / "inputs.json")
    print(f"[e11] inputs of {src} -> {dst}: {len(rec['files'])} files, sha256 checked")
    return 0


# --------------------------------------------------------------------- run --
def _classes(df: pd.DataFrame) -> dict:
    """P2-1 classes of the queries whose LCC is empty (plan section 1.2)."""
    lcc0 = df["lcc_size"] == 0
    unp = (df["lcc_status"] != "optimal") | df["pairwise_status"].isin(UNPROVEN)
    U = lcc0 & unp
    rest = lcc0 & ~U
    A = rest & (df["edgeacc_size"] == 0)
    B = rest & (df["edgeacc_size"] > 0) & (df["pairwise_size"] == 0)
    C = rest & (df["pairwise_size"] > 0)
    pw_proven_empty = (df["pairwise_size"] == 0) & df["pairwise_status"].isin(["empty", "optimal"])
    return {"lcc_empty": int(lcc0.sum()), "A": int(A.sum()), "B": int(B.sum()), "C": int(C.sum()), "U": int(U.sum()),
            "nesting_violations": int((((df["lcc_size"] > 0) & pw_proven_empty)
                                       | ((df["pairwise_size"] > 0) & (df["edgeacc_size"] == 0))
                                       | ((df["lcc_size"] > 0) & (df["edgeacc_size"] == 0))).sum())}


def summarize_run(df: pd.DataFrame) -> dict:
    nz = df[df["lcc_size"] >= 2]
    out = {"queries": int(len(df)), "lcc_nonempty": int(len(nz)),
           "lcc_nonempty_share": float(len(nz) / len(df)) if len(df) else None,
           "lcc_status": {k: int(v) for k, v in df["lcc_status"].value_counts().items()},
           "lcc_proven_share": float((df["lcc_status"] == "optimal").mean()) if len(df) else None,
           "lcc_size_median_nonempty": float(nz["lcc_size"].median()) if len(nz) else None,
           "edgeacc_nonempty": int((df["edgeacc_size"] > 0).sum()),
           "pairwise_nonempty": int((df["pairwise_size"] > 0).sum()),
           "pairwise_status": {k: int(v) for k, v in df["pairwise_status"].value_counts().items()},
           "classes": _classes(df)}
    for c in ("covisit_pair_rate", "member_meet_rate"):
        out[f"{c}_mean_nonempty"] = float(nz[c].mean()) if len(nz) else None
    return out


class _Stop(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def _requested(args, cfg: dict) -> dict:
    """The query options of a run as recorded in its summary (to compare with an existing DONE run)."""
    from lacs.apps import geo

    mode = args.mode or geo.gcfg(cfg).get("queries", "sample")
    return {"mode": mode, "n_requested": int(args.n or cfg["search"]["n_queries"]) if mode != "all" else None,
            "subset": int(args.subset or 0), "limit": int(args.limit or 0)}


def _recorded(js: dict) -> dict:
    return {"mode": js.get("mode"), "n_requested": js.get("n_requested"),
            "subset": int((js.get("subset") or {}).get("n_requested", (js.get("subset") or {}).get("n", 0)) or 0),
            "limit": int(js.get("limit") or 0)}


def _load_cfg(args):
    from lacs.config import load_config, run_paths

    s = float(args.s)
    sets = list(args.set or []) + [f"geo.theta_mult={s!r}"]
    if getattr(args, "n_proc", 0):
        sets.append(f"search.n_proc={int(args.n_proc)}")
    cfg = load_config(args.config, sets)
    return s, cfg, run_paths(cfg)


def _select(args, cfg: dict, ctx: dict, k: int, out: Path) -> tuple[list[int], dict]:
    """geo-e2's query selection (with the expected-set check), then the pilot subset / test limit."""
    from lacs.apps import geo

    g = geo.gcfg(cfg)
    mode = args.mode or g.get("queries", "sample")
    n = int(args.n or cfg["search"]["n_queries"])
    qs, _, n_elig = geo._select_queries(cfg, ctx, k, mode, n)
    qs = np.asarray(qs, dtype=np.int64)
    info = {"mode": mode, "n_requested": n if mode != "all" else None, "eligible": n_elig, "selected": int(len(qs))}
    if args.expect_queries_from:
        exp = _reference_queries(Path(args.expect_queries_from), args.expect_kind)
        same = set(exp) == set(int(q) for q in qs)
        info["expected_queries"] = {"file": args.expect_queries_from, "kind": args.expect_kind, "n": len(exp),
                                    "same": bool(same)}
        if not same:
            _dump(info, out / "e11_selection_mismatch.json")
            raise _Stop(6, f"!! query selection differs from {args.expect_queries_from}: {len(qs)} selected, "
                           f"{len(exp)} expected, {len(set(exp) ^ set(int(q) for q in qs))} differ "
                           f"(see {out / 'e11_selection_mismatch.json'})")
    if args.subset:
        rs = np.random.default_rng(int(args.subset_seed))
        qs = np.sort(rs.choice(qs, size=min(int(args.subset), len(qs)), replace=False))
        info["subset"] = {"n": int(len(qs)), "n_requested": int(args.subset), "seed": int(args.subset_seed),
                          "of": info["selected"]}
    if args.limit:
        qs = qs[: int(args.limit)]
        info["limit"] = int(args.limit)
    return [int(q) for q in qs], info


def _block_files(ctx: dict, tag: str) -> list[Path]:
    d = Path(ctx["work"]) / "blocks"
    return sorted(d.glob(f"{tag}_x_*.jsonl")) + sorted(d.glob(f"{tag}_b_*.jsonl")) if d.is_dir() else []


def cmd_run(args) -> int:
    from lacs.apps import geo
    from lacs.utils import setup_logging

    from p2 import watchdog

    s, cfg, paths = _load_cfg(args)
    rname = args.results_name
    out = paths["results"] / rname / s_label(s)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", str(out / "e11.log"))
    if (out / "DONE").exists() and not args.force:
        want, js_path = _requested(args, cfg), out / "e11_summary.json"
        have = _recorded(json.loads(js_path.read_text())) if js_path.exists() else None
        if have != want:
            print(f"!! {out} is done with other query options ({have}) than requested ({want}): "
                  f"refusing to mix runs. Stop and report; do not move or delete any result")
            return 7
        print(f"[skip] {out}: done")
        return 0
    for f in REQUIRED_INPUTS:
        if not (paths["root"] / f).is_file():
            print(f"!! {paths['root'] / f} missing (stage prepare-inputs)")
            return 2
    watchdog.install()
    k = int(cfg["search"]["k"])
    npr = int(cfg["search"].get("n_proc", 1))
    tl_x, tl_b = float(args.time_limit_exact), float(args.time_limit_baselines)
    t0 = time.time()
    tag = f"{rname}_{s_label(s)}"
    ctx = geo._e2_context(cfg, paths, tag)
    try:
        qlist, sel = _select(args, cfg, ctx, k, out)
    except _Stop as e:
        print(str(e))
        return e.code
    cached = _block_files(ctx, tag)
    if args.fresh:  # plan 4.1 timing: nothing may come from an earlier, interrupted attempt
        for f in cached:
            f.unlink()
        LOG.info("--fresh: removed %d cached engine blocks of %s", len(cached), tag)
    info = {"dataset": Path(cfg["out_dir"]).name, "s": s, "k": k, "tolerance": ctx["tol"], **sel,
            "seed": int(cfg["seed"]), "theta_median": float(np.median(ctx["theta"])), "r_global": ctx["r_global"],
            "time_limit_exact": tl_x, "time_limit_baselines": tl_b, "n_proc": npr,
            "engine_block": int(geo.gcfg(cfg).get("engine_block") or 0), "fresh": bool(args.fresh),
            "cached_blocks_found": len(cached), "cached_blocks_used": 0 if args.fresh else len(cached)}
    LOG.info("E11 %s %s: %d queries (%s, eligible %d), theta median %.4g, cached blocks %d%s", info["dataset"],
             s_label(s), len(qlist), info["mode"], info["eligible"], info["theta_median"], len(cached),
             " (removed)" if args.fresh else "")
    t1 = time.time()
    rows_x = geo._run_blocks(ctx, qlist, ["exact"], k, f"{tag}_x", theta="acc", time_limit=tl_x, r=ctx["r_global"],
                             n_proc=npr)
    t2 = time.time()
    rows_b = geo._run_blocks(ctx, qlist, ["edgeacc", "pairwise_exact"], k, f"{tag}_b", theta="acc", time_limit=tl_b,
                             members=False, r=ctx["r_global"], n_proc=npr)
    t3 = time.time()
    xs = {int(r["q"]): r for r in rows_x}
    bs = {(int(r["q"]), r["algo"]): r for r in rows_b}
    if len(xs) != len(qlist) or len(bs) != 2 * len(qlist):
        raise RuntimeError(f"engine returned {len(xs)} / {len(bs)} rows for {len(qlist)} queries")
    recs = []
    for q in qlist:
        x, ea, pw = xs[q], bs[(q, "edgeacc")], bs[(q, "pairwise_exact")]
        H = np.asarray(x.get("members") or [], dtype=np.int64)
        if int(x["size"]) != len(H):
            raise RuntimeError(f"query {q}: size {x['size']} but {len(H)} members")
        met = geo._member_metrics(ctx, H, True) if len(H) >= 2 else {}
        recs.append({"q": q, "s": s, "k": k,
                     "lcc_status": x["status"], "lcc_size": int(x["size"]), "lcc_ub": x.get("ub"),
                     "lcc_time_ms": x.get("time_ms"), "lcc_ub_ms": x.get("ub_ms"), "lcc_tau": x.get("tau"),
                     "lcc_feasible": x.get("feasible"),
                     **{c: met.get(c, np.nan) for c in METRICS},
                     "edgeacc_status": ea["status"], "edgeacc_size": int(ea["size"]),
                     "pairwise_status": pw["status"], "pairwise_size": int(pw["size"]), "pairwise_ub": pw.get("ub"),
                     "pairwise_time_ms": pw.get("time_ms")})
    df = pd.DataFrame(recs)
    t4 = time.time()
    summ = summarize_run(df)
    timing = {"setup_s": t1 - t0, "engine_exact_s": t2 - t1, "engine_baselines_s": t3 - t2, "metrics_s": t4 - t3,
              "total_s": t4 - t0}
    _write_csv_atomic(df, out / "e11_per_query.csv")
    from lacs.utils import dump_json

    dump_json({k_: v for k_, v in cfg.items() if not k_.startswith("_")}, out / "config_resolved.json")
    _dump({**info, "timing": timing, **summ, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, out / "e11_summary.json")
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[e11] {info['dataset']} {s_label(s)}: {len(df)} queries, LCC non-empty {summ['lcc_nonempty']}, "
          f"proven {summ['lcc_status'].get('optimal', 0)}, {timing['total_s']:.0f} s -> {out}")
    return 0


# ------------------------------------------------------- P2-4b group export --
def cmd_export_groups(args) -> int:
    """The LCC groups of a finished E11 run as P0's group export, read back from the run's cached exact blocks
    (the same geo._run_blocks call as the run, with the engine disabled: a missing block is an error)."""
    from lacs.apps import geo
    from lacs.utils import setup_logging

    from p0kit import bridge
    from p0kit.export_groups import GEO_METRICS
    from p0kit.formats import JsonlWriter

    s, cfg, paths = _load_cfg(args)
    rname = args.results_name
    run_out = paths["results"] / rname / s_label(s)
    dst = Path(args.out)
    if (dst / "manifest.json").exists() and not args.force:
        have = json.loads((dst / "manifest.json").read_text()).get("query_options")
        if have != _requested(args, cfg):
            print(f"!! {dst} was exported with other query options ({have}) than requested "
                  f"({_requested(args, cfg)}). Stop and report; do not move or delete any result")
            return 7
        print(f"[skip] {dst}: export done")
        return 0
    if not (run_out / "DONE").exists():
        print(f"!! {run_out}: the E11 run of this s has not finished (needed by the export)")
        return 2
    js = json.loads((run_out / "e11_summary.json").read_text())
    want = _requested(args, cfg)
    if _recorded(js) != want:
        print(f"!! {run_out} was run with other query options ({_recorded(js)}) than requested ({want})")
        return 7
    dst.mkdir(parents=True, exist_ok=True)
    setup_logging("INFO", str(dst / "export.log"))
    t0 = time.time()
    k = int(cfg["search"]["k"])
    tag = f"{rname}_{s_label(s)}"
    ctx = geo._e2_context(cfg, paths, tag)
    try:
        qlist, _ = _select(args, cfg, ctx, k, dst)
    except _Stop as e:
        print(str(e))
        return e.code

    def _no_engine(*a, **kw):
        raise RuntimeError("export-groups: an exact block of the E11 run is not cached (rerun the E11 stage)")

    ctx["cs"].run = _no_engine  # instance attribute: only this context, the engine is never started
    rows = geo._run_blocks(ctx, qlist, ["exact"], k, f"{tag}_x", theta="acc", time_limit=float(js["time_limit_exact"]),
                           r=ctx["r_global"], n_proc=int(js.get("n_proc") or 1))
    xs = {int(r["q"]): r for r in rows}
    per_q = pd.read_csv(run_out / "e11_per_query.csv").set_index("q")
    if set(xs) != set(int(q) for q in per_q.index):
        raise RuntimeError("cached exact rows and e11_per_query.csv cover different queries")
    w = JsonlWriter(str(dst / "groups.jsonl.gz"), append=False)
    recs = []
    name = Path(cfg["out_dir"]).name
    for q in qlist:
        x = xs[q]
        H = [int(v) for v in (x.get("members") or [])]
        if len(H) != int(x["size"]) or len(H) != int(per_q.at[q, "lcc_size"]):
            raise RuntimeError(f"query {q}: cached size {x['size']}, {len(H)} members, E11 size {per_q.at[q, 'lcc_size']}")
        g_rec = {"dataset": name, "setting": "default", "query": q, "k": k, "method": "lcc_exact",
                 "members": H if len(H) <= int(args.members_max) else None, "size": len(H), "status": x["status"],
                 "radius_km": None}
        for m in GEO_METRICS:
            v = per_q.at[q, m] if m in per_q.columns else None
            g_rec[m] = None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)
        w.write_many([g_rec])
        recs.append({"q": q, "method": "lcc_exact", "size": len(H), "status": x["status"]})
    w.close()
    pd.DataFrame(recs).to_csv(dst / "per_query.csv", index=False)
    ds = bridge.load_geo(cfg["out_dir"], tolerance=ctx["tol"])  # users.csv: the unscaled theta_q50
    inst = bridge.save_instance(dst, ds, tag="default", meta={"tolerance": ctx["tol"], "units": "km",
                                                            "search_theta_mult": s})
    man = {"dataset": name, "config": str(Path(args.config).resolve()), "s": s, "k": k, "rows": len(recs),
           "from_run": str(run_out), "query_options": want, "instance": inst, "engine_searches": 0,
           "engine_note": "groups read from the E11 cache; only the context's core-number call ran the engine binary",
           "seconds": time.time() - t0, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    _dump(man, dst / "manifest.json")
    print(f"[e11 export] {name} {s_label(s)}: {len(recs)} groups from the cached E11 run -> {dst}")
    return 0


# ------------------------------------------------------------ references --
def _reference_rows(path: Path, kind: str) -> pd.DataFrame:
    """LCC rows (q, status, size) of an earlier run: 'r4main' / 'e5main' = geo_e2_per_query.csv (all queries),
    'r4sweep' / 'e5sweep' = the theta_q50 run of the tolerance parameter study in geo_e2_sweep_per_query.csv."""
    if kind in ("r4main", "e5main"):
        d = pd.read_csv(path, usecols=["q", "method", "status", "size"])
    elif kind in ("r4sweep", "e5sweep"):
        d = pd.read_csv(path, usecols=["sweep", "k", "tolerance", "q", "method", "status", "size"])
        d = d[(d["sweep"] == "tolerance") & (d["tolerance"] == "theta_q50") & (d["k"] == 3)]
    else:
        raise ValueError(f"unknown reference kind {kind}")
    d = d[d["method"] == "lcc_exact"][["q", "status", "size"]]
    if d["q"].duplicated().any():
        raise ValueError(f"{path}: duplicated queries in the {kind} rows")
    return d.set_index("q")


def _reference_queries(path: Path, kind: str) -> list[int]:
    return [int(q) for q in _reference_rows(path, kind).index]


def cmd_gate(args) -> int:
    """Plan section 4.2: at s = 1, every query proven optimal in both runs has the same LCC size."""
    mine = pd.read_csv(args.run).set_index("q")
    ref = _reference_rows(Path(args.ref), args.ref_kind)
    common = mine.index.intersection(ref.index)
    both = [q for q in common if mine.at[q, "lcc_status"] == "optimal" and ref.at[q, "status"] == "optimal"]
    mism = [q for q in both if int(mine.at[q, "lcc_size"]) != int(ref.at[q, "size"])]
    res = {"run": str(args.run), "ref": str(args.ref), "ref_kind": args.ref_kind, "queries_run": int(len(mine)),
           "queries_ref": int(len(ref)), "common": int(len(common)), "missing_in_ref": int(len(mine) - len(common)),
           "both_proven": len(both), "mismatches": len(mism),
           "examples": [{"q": int(q), "run": int(mine.at[q, "lcc_size"]), "ref": int(ref.at[q, "size"])}
                        for q in mism[:20]],
           "proven_only_run": int(sum(1 for q in common if mine.at[q, "lcc_status"] == "optimal"
                                      and ref.at[q, "status"] != "optimal")),
           "proven_only_ref": int(sum(1 for q in common if mine.at[q, "lcc_status"] != "optimal"
                                      and ref.at[q, "status"] == "optimal")),
           "nonempty_run_vs_ref_common": [int((mine.loc[common, "lcc_size"] > 0).sum()),
                                          int((ref.loc[common, "size"] > 0).sum())]}
    res["pass"] = bool(len(mism) == 0 and len(both) > 0 and res["missing_in_ref"] == 0)
    _dump(res, Path(args.out))
    print(f"[gate] {args.run}: {len(both)} queries proven in both, {len(mism)} mismatches, "
          f"{res['missing_in_ref']} not in the reference -> {'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 4


# ------------------------------------------------------------------- pilot --
def cmd_decide(args) -> int:
    """Plan section 4.1: the pilot (200 queries per dataset at s = 3) projects the whole E11 run as if every
    multiplier cost as much as s = 3 (an upper bound: smaller s are cheaper): per dataset and multiplier, the
    pilot's fixed set-up time (inputs, core numbers, query selection) plus its per-query time (engine passes
    and metrics, divided by the pilot's queries) times the dataset's queries; summed over the 8 multipliers.
    Projection above --threshold-h: Gowalla falls back to the 2,000 queries of round 4's tolerance study (rule
    fixed in the plan). Projection still above --stop-h: stop and wait for the user (exit code 5)."""
    pil = {}
    for spec in args.pilot:
        name, d = spec.split("=", 1)
        js = json.loads((Path(d) / "e11_summary.json").read_text())
        if not js.get("fresh") or int(js.get("cached_blocks_used", 1)) != 0:
            print(f"!! pilot {name} ({d}) was not a fresh run (cached engine blocks would make its time too low): "
                  f"stop and report")
            return 7
        n_sub = js["subset"]["n"] if js.get("subset") else js["selected"]
        tm = js["timing"]
        setup = float(tm.get("setup_s", 0.0))
        pil[name] = {"pilot_seconds": tm["total_s"], "setup_seconds": setup,
                     "per_query_seconds": (tm["total_s"] - setup) / max(n_sub, 1), "pilot_queries": n_sub,
                     "full_queries": js.get("subset", {}).get("of", js["selected"])}
    n_s = len(S_GRID)

    def project(full: dict) -> float:
        return sum((p["setup_seconds"] + p["per_query_seconds"] * full[name]) * n_s for name, p in pil.items()) / 3600

    full = {name: p["full_queries"] for name, p in pil.items()}
    h_all = project(full)
    dec = {"grid": S_GRID, "pilot": pil, "projection_h_full": h_all, "threshold_h": args.threshold_h,
           "stop_h": args.stop_h, "gowalla_mode": "all", "fallback_queries": args.fallback_n}
    h = h_all
    if h_all > args.threshold_h and "gowalla" in full:
        full2 = dict(full, gowalla=min(int(args.fallback_n), full["gowalla"]))
        h = project(full2)
        dec.update(gowalla_mode="r4_sample", projection_h_fallback=h)
    dec["projection_h_used"] = h
    dec["stop"] = bool(h > args.stop_h)
    dec["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _dump(dec, Path(args.out))
    line = (f"pilot: projected E11 total {h_all:.1f} h with every query"
            + (f"; Gowalla falls back to {args.fallback_n} queries (rule of plan 4.1): {h:.1f} h"
               if dec["gowalla_mode"] != "all" else "")
            + (f" -> above {args.stop_h:g} h: STOP, waiting for the user" if dec["stop"] else " -> OK"))
    print(line)
    return 5 if dec["stop"] else 0


# ----------------------------------------------------------------- summary --
def cmd_summary(args) -> int:
    root = Path(args.root)
    rows = []
    for js in sorted(root.glob(f"*/results/{args.results_name}/s*/e11_summary.json")):
        d = json.loads(js.read_text())
        c = d.get("classes", {})
        rows.append({"dataset": d["dataset"], "s": d["s"], "queries": d["queries"], "mode": d["mode"],
                     "lcc_nonempty_share": d["lcc_nonempty_share"], "lcc_proven_share": d["lcc_proven_share"],
                     "lcc_size_median_nonempty": d["lcc_size_median_nonempty"],
                     "covisit_pair_rate_mean_nonempty": d.get("covisit_pair_rate_mean_nonempty"),
                     "member_meet_rate_mean_nonempty": d.get("member_meet_rate_mean_nonempty"),
                     "A": c.get("A"), "B": c.get("B"), "C": c.get("C"), "U": c.get("U"),
                     "nesting_violations": c.get("nesting_violations"),
                     "minutes": d["timing"]["total_s"] / 60, "done": (js.parent / "DONE").exists()})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("!! no E11 results found")
        return 1
    df = pd.DataFrame(rows).sort_values(["dataset", "s"])
    df.to_csv(out / "e11_summary.csv", index=False)
    gates = {p.parent.parent.parent.name: json.loads(p.read_text())
             for p in sorted(root.glob(f"*/results/{args.results_name}/gate_s1.json"))}
    lines = ["# E11 summary (P2-4)", "", "Gate (s = 1 against round 4 / E5): " +
             ", ".join(f"{k}: {'PASS' if v.get('pass') else 'FAIL'} ({v.get('both_proven')} proven in both, "
                       f"{v.get('mismatches')} mismatches)" for k, v in gates.items()), "",
             "| " + " | ".join(df.columns) + " |", "|" + " --- |" * len(df.columns)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in r) + " |")
    viol = int(df["nesting_violations"].fillna(0).sum())
    lines += ["", f"nesting violations over all runs: {viol} (must be 0)"]
    (out / "E11_SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if viol == 0 else 1


# ------------------------------------------------------------- make-config --
def cmd_make_config(args) -> int:
    """Write a config file = base config + --set overrides (for tools that take no --set, e.g. the P0 export)."""
    import yaml

    from lacs.config import apply_overrides

    base = yaml.safe_load(Path(args.base).read_text()) or {}
    cfg = apply_overrides(base, args.set)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[e11] config {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m p2.e11", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare-inputs")
    a.add_argument("--src", required=True)
    a.add_argument("--dst", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--s", type=float, required=True)
    r.add_argument("--mode", choices=["all", "sample"], default=None, help="default: the config's geo.queries")
    r.add_argument("--n", type=int, default=None, help="sample size (default: the config's search.n_queries)")
    r.add_argument("--subset", type=int, default=0, help="random subset of the selected queries (pilot)")
    r.add_argument("--subset-seed", type=int, default=20261027)
    r.add_argument("--limit", type=int, default=0, help="first N queries only (tests)")
    r.add_argument("--expect-queries-from", default=None, help="CSV whose LCC query set the selection must equal")
    r.add_argument("--expect-kind", default="r4sweep", choices=["r4main", "e5main", "r4sweep", "e5sweep"])
    r.add_argument("--results-name", default="e11")
    r.add_argument("--time-limit-exact", type=float, default=60.0)
    r.add_argument("--time-limit-baselines", type=float, default=10.0)
    r.add_argument("--n-proc", type=int, default=0)
    r.add_argument("--set", nargs="*", default=[])
    r.add_argument("--force", action="store_true")
    r.add_argument("--fresh", action="store_true", help="delete this run's cached engine blocks first (pilot timing)")
    ex = sub.add_parser("export-groups")
    ex.add_argument("--config", required=True)
    ex.add_argument("--s", type=float, required=True)
    ex.add_argument("--mode", choices=["all", "sample"], default=None)
    ex.add_argument("--n", type=int, default=None)
    ex.add_argument("--expect-queries-from", default=None)
    ex.add_argument("--expect-kind", default="r4sweep", choices=["r4main", "e5main", "r4sweep", "e5sweep"])
    ex.add_argument("--results-name", default="e11")
    ex.add_argument("--members-max", type=int, default=5000)
    ex.add_argument("--set", nargs="*", default=[])
    ex.add_argument("--out", required=True)
    ex.add_argument("--force", action="store_true")
    gte = sub.add_parser("gate")
    gte.add_argument("--run", required=True)
    gte.add_argument("--ref", required=True)
    gte.add_argument("--ref-kind", required=True, choices=["r4main", "e5main", "r4sweep", "e5sweep"])
    gte.add_argument("--out", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--pilot", nargs="+", required=True, help="name=pilot result dir (with e11_summary.json)")
    d.add_argument("--threshold-h", type=float, default=12.0)
    d.add_argument("--stop-h", type=float, default=24.0)
    d.add_argument("--fallback-n", type=int, default=2000)
    d.add_argument("--out", required=True)
    sm = sub.add_parser("summary")
    sm.add_argument("--root", required=True)
    sm.add_argument("--out", required=True)
    sm.add_argument("--results-name", default="e11")
    mc = sub.add_parser("make-config")
    mc.add_argument("--base", required=True)
    mc.add_argument("--out", required=True)
    mc.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    if args.cmd == "export-groups":  # same selection code as run: no subset, no limit
        args.subset, args.subset_seed, args.limit = 0, 20261027, 0
    fn = {"prepare-inputs": cmd_prepare_inputs, "run": cmd_run, "gate": cmd_gate, "decide": cmd_decide,
          "summary": cmd_summary, "make-config": cmd_make_config, "export-groups": cmd_export_groups}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
