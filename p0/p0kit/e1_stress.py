"""E1: interaction stress test of the groups returned by each method.

  python -m p0kit.e1_stress --config configs/e1_stress.yaml \
      --instance runs_p0/livejournal/e2/sigma0.4/instance.json \
      --groups runs_p0/livejournal/e2/sigma0.4/groups.jsonl.gz \
      --out runs_p0/livejournal/e1/sigma0.4 --workers 8

Works with any instance/groups pair in the formats of p0kit/formats.py, so the
groups exported by the LACS engine (e.g. the Gowalla/Brightkite meetup groups)
can be stressed the same way.

Output: e1_rows.jsonl.gz, one row per (query, method, variant) with the static
group statistics and the run aggregates of p0kit.dynamics.aggregate; plus
variants.csv and manifest.json. Re-running resumes: groups already present in
e1_rows.jsonl.gz are skipped.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np

from .dynamics import GroupContext, Variant, aggregate, run_variant, stable_seed, variant_table
from .formats import Instance, JsonlWriter, read_groups, recover_jsonl, rewrite_jsonl
from .runner_common import host_info, load_yaml, run_pool, write_json


def _scalar_extras(g: Dict) -> Dict:
    out = {}
    for k, v in g.items():
        if k in ("dataset", "query", "k", "method", "members", "setting"):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[f"g_{k}"] = v
    return out


def _work(shared: Dict, g: Dict):
    inst: Instance = shared["inst"]
    cfg = shared["cfg"]
    variants = shared["variants"]
    base = {"dataset": g.get("dataset", shared["dataset"]), "setting": g.get("setting", shared["setting"]),
            "query": g["query"], "k": g.get("k"), "method": g["method"]}
    extra = _scalar_extras(g)
    if g.get("members") is None:
        return [{**base, **extra, "size": g.get("size"), "variant": "*", "skipped": "members_not_stored"}]
    members = np.asarray(g["members"], dtype=np.int64)
    n = len(members)
    if n < 2:
        return [{**base, **extra, "size": n, "variant": "*", "skipped": "size<2"}]
    if n > cfg.get("max_group", 5000):
        return [{**base, **extra, "size": n, "variant": "*", "skipped": "size>max_group"}]
    seed0 = stable_seed(cfg.get("seed", 0), base["dataset"], base["setting"], base["query"], base["method"])
    t0 = time.perf_counter()
    ctx = GroupContext(inst.indptr, inst.indices, inst.X, inst.theta, members, shared.get("pos"),
                       need_ext=shared["need_ext"], ext_max_neighbors=cfg.get("ext_max_neighbors", 200), seed=seed0,
                       need_open=shared["need_open"])
    rows = []
    for var in variants:
        row = {**base, **extra, **ctx.stats, "variant": var.name}
        if n > var.max_n:
            row["skipped"] = "size>max_n"
        else:
            row.update(aggregate(run_variant(ctx, var, stable_seed(seed0, var.name))))
            row["skipped"] = ""
        rows.append(row)
    rows[0]["group_seconds"] = time.perf_counter() - t0
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m p0kit.e1_stress")
    ap.add_argument("--config", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--methods", default=None, help="comma separated subset")
    ap.add_argument("--limit", type=int, default=0, help="only the first N groups (smoke tests)")
    ap.add_argument("--dataset", default="", help="dataset name if the groups file has none")
    ap.add_argument("--setting", default="", help="setting name if the groups file has none")
    ap.add_argument("--lcc-only", action="store_true",
                    help="keep only queries whose lcc_exact group has >= 2 members (the paired comparison)")
    ap.add_argument("--other-sample", type=int, default=0,
                    help="with --lcc-only, also keep this many randomly chosen other queries")
    ap.add_argument("--max-lcc-queries", type=int, default=0,
                    help="with --lcc-only, keep a random sample of at most this many queries with an LCC (0: all)")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    variants = [Variant.from_dict(v, cfg.get("defaults")) for v in cfg["variants"]]
    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "e1_rows.jsonl.gz")
    done = set()
    if os.path.exists(rows_path):
        # resume: a group counts as done only if all its rows were written (a run killed while
        # appending leaves a truncated file and possibly a partial group; both are repaired)
        recs, info = recover_jsonl(rows_path)
        names = {v.name for v in variants}
        seen = {}
        for r in recs:
            key = (str(r.get("dataset")), str(r.get("setting")), r["query"], r["method"])
            seen.setdefault(key, set()).add(r["variant"])
        done = {key for key, vs in seen.items() if "*" in vs or names <= vs}
        keep = [r for r in recs if (str(r.get("dataset")), str(r.get("setting")), r["query"], r["method"]) in done]
        if info["truncated"] or info["bad_lines"] or len(keep) != len(recs):
            rewrite_jsonl(rows_path, keep)
            print(f"[e1] repaired {rows_path} after an interrupted run: kept {len(keep)} of {len(recs)} rows "
                  f"({len(done)} complete groups)")
        if os.path.exists(os.path.join(args.out, "DONE")) and len(done) < len(seen):
            os.remove(os.path.join(args.out, "DONE"))
    groups = []
    methods = set(args.methods.split(",")) if args.methods else None
    keep_q = None
    if args.lcc_only:
        lcc_q, all_q = set(), set()
        for g in read_groups(args.groups):
            all_q.add(g["query"])
            if g["method"] == "lcc_exact" and (g.get("size") or len(g.get("members") or [])) >= 2:
                lcc_q.add(g["query"])
        rng = np.random.default_rng(cfg.get("seed", 0))
        lcc_sorted = sorted(lcc_q)
        if args.max_lcc_queries and len(lcc_sorted) > args.max_lcc_queries:
            pick = rng.choice(len(lcc_sorted), size=args.max_lcc_queries, replace=False)
            keep_q = {lcc_sorted[i] for i in pick}
        else:
            keep_q = set(lcc_sorted)
        n_lcc_kept = len(keep_q)
        others = sorted(all_q - lcc_q)
        if args.other_sample and others:
            pick = rng.choice(len(others), size=min(args.other_sample, len(others)), replace=False)
            keep_q |= {others[i] for i in pick}
        print(f"[e1] queries: {n_lcc_kept} of {len(lcc_q)} with an LCC kept, {len(keep_q) - n_lcc_kept} others kept, "
              f"of {len(all_q)}")
    for g in read_groups(args.groups):
        if methods and g["method"] not in methods:
            continue
        if keep_q is not None and g["query"] not in keep_q:
            continue
        key = (str(g.get("dataset", args.dataset)), str(g.get("setting", args.setting)), g["query"], g["method"])
        if key in done:
            continue
        groups.append(g)
        if args.limit and len(groups) >= args.limit:
            break
    # big groups first for load balance
    groups.sort(key=lambda g: -(len(g["members"]) if g.get("members") is not None else 0))
    inst = Instance(args.instance, mmap=True)
    shared = {"inst": inst, "cfg": cfg, "variants": variants, "dataset": args.dataset, "setting": args.setting,
              "need_ext": any(v.lambda_ext > 0 for v in variants),
              "need_open": any(v.open for v in variants),
              "pos": np.full(inst.n, -1, dtype=np.int32)}
    write_json(os.path.join(args.out, "manifest.json"), {
        "instance": os.path.abspath(args.instance), "groups": os.path.abspath(args.groups),
        "n_groups_todo": len(groups), "n_groups_done_before": len(done), "config": cfg, "host": host_info(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    import pandas as pd

    pd.DataFrame(variant_table(variants)).to_csv(os.path.join(args.out, "variants.csv"), index=False)
    w = JsonlWriter(rows_path, append=True)
    t0 = time.time()
    run_pool(shared, _work, groups, args.workers, log_every=200, label="e1", on_result=w.write_many)
    w.close()
    open(os.path.join(args.out, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[done] {args.out}: {len(groups)} groups in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
