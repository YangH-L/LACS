"""Open-group sweep (E1 follow-up) and E6 dynamics on the groups returned by each method.

  PYTHONPATH=p0:. python -m e1x.open_sweep --config e1x/configs/e1_open_meetup.yaml \
      --instance runs_p0/gowalla/e1_input/instance.json \
      --groups runs_p0/gowalla/e1_input/groups.jsonl.gz \
      --out runs/e1_open/gowalla --lcc-only --workers 12

Plans: docs/PLANS.md (lacs_e5_prereg_draft_20261025) section 8 (meetup groups exported by P0) and
docs/PLANS.md (lacs_e6_plan_20261025) section 7 (Congress groups). Uses the P0 kernels unchanged
(p0/p0kit, kit digest dc25c9f4...) and P0's settings.

Per group (queries whose lcc_exact group has >= 2 members, as P0's --lcc-only): the static open
conditions of e1x.conditions on the same outside-neighbour lists the dynamics use (P0's sampling:
at most ext_max_neighbors per member, same seeds, so w = 1 reproduces P0's hk_open_G), then every
variant of the config:
  closed P0 variants   name/model/graph/weights as in p0kit.dynamics.Variant (hk, follow, dw, ...)
  open, equal inside   {name, open_w: w}: members weigh themselves and accepted in-group
                       neighbours 1 and each accepted outside neighbour w
  observed weights     {name, observed: true[, open: true]}: W[v,u] = weight of the directed
                       Twitter edge u->v (v engaged with u) / mean edge weight, self weight 1;
                       needs --weights (npz with src, dst, w over vertex ids)
Output in --out: e1x_rows.jsonl.gz (one row per group and variant), e1x_groups.jsonl.gz (one row
per group: static conditions, first open w with a broken interaction), manifest.json; then
e1x.summarize writes e1x_summary.csv. Re-running resumes (finished groups are skipped).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "p0", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from p0kit import ffi  # noqa: E402
from p0kit.dynamics import DEFAULTS, GroupContext, Variant, aggregate, run_variant, stable_seed  # noqa: E402
from p0kit.formats import Instance, JsonlWriter, read_groups, read_jsonl, recover_jsonl, rewrite_jsonl  # noqa: E402
from p0kit.runner_common import host_info, kit_digest, load_yaml, run_pool, write_json  # noqa: E402

from e1x.conditions import open_conditions  # noqa: E402

VERSION = "e1x-0.1"


def outside_lists(gptr, gidx, members, max_neighbors: int, rng, pos_buf):
    """Each member's outside neighbours (vertex ids), sampled exactly like
    p0kit.graphs.outside_neighbors (same rng calls in the same order)."""
    members = np.asarray(members, dtype=np.int64)
    pos_buf[members] = np.arange(len(members), dtype=np.int32)
    picks = []
    try:
        for v in members:
            nb = np.asarray(gidx[gptr[v]:gptr[v + 1]], dtype=np.int64)
            out = nb[pos_buf[nb] < 0]
            if max_neighbors and len(out) > max_neighbors:
                out = np.sort(rng.choice(out, size=max_neighbors, replace=False))
            picks.append(out)
    finally:
        pos_buf[members] = -1
    optr = np.zeros(len(members) + 1, np.int64)
    optr[1:] = np.cumsum([len(p) for p in picks])
    oid = np.concatenate(picks) if optr[-1] else np.zeros(0, np.int64)
    return optr, oid


class ObservedWeights:
    """W[v, u] for the directed engagement u -> v (v listens to u), normalised by the mean weight."""

    def __init__(self, path: str, n: int):
        z = np.load(path)
        src, dst, w = z["src"].astype(np.int64), z["dst"].astype(np.int64), z["w"].astype(np.float64)
        ok = (w > 0) & (src != dst)
        # normaliser: 'norm_mean' when the file gives one (E6: mean over all network edges, plan section 3),
        # else the mean of the weights in the file
        self.mean = float(z["norm_mean"]) if "norm_mean" in z.files else float(w[ok].mean())
        self.n = n
        self.key = dst[ok] * n + src[ok]  # (listener, speaker)
        o = np.argsort(self.key)
        self.key, self.val = self.key[o], w[ok][o] / self.mean

    def lookup(self, listener: np.ndarray, speaker: np.ndarray) -> np.ndarray:
        k = np.asarray(listener, np.int64) * self.n + np.asarray(speaker, np.int64)
        i = np.searchsorted(self.key, k)
        i = np.minimum(i, len(self.key) - 1)
        return np.where(self.key[i] == k, self.val[i], 0.0) if len(self.key) else np.zeros(len(k))


def _sim(ctx: GroupContext, P: dict, W_self=None, W_edge=None, open_ptr=None, open_X=None, open_W=None,
         model="hk", seed=0):
    t0 = time.perf_counter()
    de = max(1, min(int(P["diam_every"]) * max(1, int(np.ceil((ctx.n / 1000.0) ** 2))), int(P["T_max"])))
    res = ffi.sim_sync(ctx.Xh, ctx.th, ctx.th, ctx.ptr, ctx.idx, W_self=W_self, W_edge=W_edge, open_ptr=open_ptr,
                       open_X=open_X, open_W=open_W, model=model, T_max=int(P["T_max"]),
                       move_tol_rel=P["move_tol_rel"], rel_eps=P["rel_eps"], diam_every=de, cons_rel=P["cons_rel"],
                       cluster_rel=P["cluster_rel"], seed=seed)
    res["sim_seconds"] = time.perf_counter() - t0
    res["n"] = ctx.n
    return res


def _work(shared: dict, g: dict):
    """One group; an unexpected error is recorded as a skipped group (counted by e1x.summarize) instead
    of ending the whole sweep."""
    try:
        return _work_group(shared, g)
    except Exception as e:  # noqa: BLE001
        import traceback
        base = {"dataset": g.get("dataset", shared["dataset"]), "setting": g.get("setting", shared["setting"]),
                "query": g["query"], "k": g.get("k"), "method": g["method"], "size": g.get("size"),
                "variant": "*", "skipped": f"error: {type(e).__name__}: {e}"}
        print(f"[e1x] ERROR on query {g['query']} {g['method']}: {e}\n{traceback.format_exc()}", flush=True)
        return [base], None


def _work_group(shared: dict, g: dict):
    inst, cfg = shared["inst"], shared["cfg"]
    base = {"dataset": g.get("dataset", shared["dataset"]), "setting": g.get("setting", shared["setting"]),
            "query": g["query"], "k": g.get("k"), "method": g["method"]}
    members = g.get("members")
    if members is None:
        return [{**base, "size": g.get("size"), "variant": "*", "skipped": "members_not_stored"}], None
    members = np.asarray(members, dtype=np.int64)
    n = len(members)
    if n < 2 or n > int(cfg.get("max_group", 5000)):
        return [{**base, "size": n, "variant": "*", "skipped": "size<2" if n < 2 else "size>max_group"}], None
    t0 = time.perf_counter()
    seed0 = stable_seed(cfg.get("seed", 0), base["dataset"], base["setting"], base["query"], base["method"])
    pos_buf = shared["pos"]
    ctx = GroupContext(inst.indptr, inst.indices, inst.X, inst.theta, members, pos_buf, need_ext=False,
                       seed=seed0, need_open=False)
    # outside neighbours with P0's sampling: GroupContext(need_open=True) draws them from default_rng(seed0)
    optr, oid = outside_lists(inst.indptr, inst.indices, members, int(cfg.get("ext_max_neighbors", 200)),
                              np.random.default_rng(seed0), pos_buf)
    oX = np.asarray(inst.X[oid], dtype=np.float64).reshape(len(oid), inst.X.shape[1])
    cond = open_conditions(ctx.Xh, ctx.th, optr, oX, float(cfg.get("margin", 1e-6)))
    grow = {**base, **ctx.stats, **cond, "n_outside_nbrs_used": int(len(oid))}
    P = {**DEFAULTS, **(cfg.get("defaults") or {})}
    obs = shared.get("obs")
    rows = []
    first_w = None
    for spec in shared["variants"]:
        name = spec["name"]
        row = {**base, "size": n, "variant": name, "w": spec.get("open_w", 0.0 if not spec.get("open") else None)}
        if spec.get("kind") == "p0":
            if n > spec["variant"].max_n:  # as P0's e1_stress
                row["skipped"] = "size>max_n"
                rows.append(row)
                continue
            res = run_variant(ctx, spec["variant"], stable_seed(seed0, name))
        elif spec.get("kind") == "open_w":
            w = float(spec["open_w"])
            res = [_sim(ctx, P, open_ptr=optr, open_X=oX, open_W=np.full(len(oid), w))]
        else:  # observed weights
            if obs is None:
                raise RuntimeError(f"variant {name} needs --weights")
            src = np.repeat(np.arange(n), np.diff(ctx.ptr))
            W_edge = obs.lookup(members[src], members[ctx.idx.astype(np.int64)])
            kw = {}
            if spec.get("open"):
                owner = np.repeat(np.arange(n), np.diff(optr))
                kw = {"open_ptr": optr, "open_X": oX, "open_W": obs.lookup(members[owner], oid)}
            res = [_sim(ctx, P, W_self=np.ones(n), W_edge=W_edge, **kw)]
        agg = aggregate(res)
        row.update(agg)
        row["broken_any"] = bool(agg["p_broken"] > 0)
        w_here = (float(spec["open_w"]) if spec.get("kind") == "open_w"
                  else (0.0 if name == cfg.get("closed_reference", "hk_eq_G") else None))
        if w_here is not None and row["broken_any"]:  # the grid of the plan: w = 0 (closed) and the open weights
            first_w = w_here if first_w is None else min(first_w, w_here)
        rows.append(row)
    grow["first_open_w_broken"] = first_w
    grow["group_seconds"] = time.perf_counter() - t0
    return rows, grow


def _variants(cfg: dict) -> list:
    out = []
    for v in cfg["variants"]:
        v = dict(v)
        if "open_w" in v:
            out.append({"name": v["name"], "kind": "open_w", "open_w": float(v["open_w"])})
        elif v.get("observed"):
            out.append({"name": v["name"], "kind": "observed", "open": bool(v.get("open", False))})
        else:
            var = Variant.from_dict(v, cfg.get("defaults"))
            if var.open or var.lambda_ext > 0:
                raise ValueError(f"{var.name}: use open_w / observed for open groups")
            out.append({"name": var.name, "kind": "p0", "variant": var, "open": False})
    names = [v["name"] for v in out]
    if len(set(names)) != len(names):
        raise ValueError("duplicate variant names")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e1x.open_sweep", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="npz with src, dst, w (observed-weight variants)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--lcc-only", action="store_true", help="queries whose lcc_exact group has >= 2 members")
    ap.add_argument("--queries-from", default=None,
                    help="only the queries present in this rows file (e.g. P0's e1_rows.jsonl.gz: P0's own sample of "
                         "1,000 queries with an LCC plus 300 without)")
    ap.add_argument("--methods", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--setting", default="")
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    variants = _variants(cfg)
    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "e1x_rows.jsonl.gz")
    grp_path = os.path.join(args.out, "e1x_groups.jsonl.gz")
    done = set()
    if os.path.exists(grp_path):  # resume: a group is done when its group row exists (written last)
        grecs, ginfo = recover_jsonl(grp_path)
        done = {(str(r["dataset"]), str(r["setting"]), r["query"], r["method"]) for r in grecs}
        recs, info = recover_jsonl(rows_path) if os.path.exists(rows_path) else ([], {"truncated": False,
                                                                                      "bad_lines": 0})
        keep = [r for r in recs if (str(r["dataset"]), str(r["setting"]), r["query"], r["method"]) in done]
        if info["truncated"] or info["bad_lines"] or len(keep) != len(recs):
            rewrite_jsonl(rows_path, keep)
        if ginfo["truncated"] or ginfo["bad_lines"]:
            rewrite_jsonl(grp_path, grecs)
        print(f"[e1x] resuming: {len(done)} groups already done")
    methods = set(args.methods.split(",")) if args.methods else None
    keep_q = None
    if args.lcc_only:
        keep_q = {g["query"] for g in read_groups(args.groups)
                  if g["method"] == "lcc_exact" and (g.get("size") or len(g.get("members") or [])) >= 2}
    if args.queries_from:
        qf = {r["query"] for r in read_jsonl(args.queries_from)}
        keep_q = qf if keep_q is None else (keep_q & qf)
        print(f"[e1x] {len(keep_q)} queries taken from {args.queries_from}")
    todo = []
    for g in read_groups(args.groups):
        if methods and g["method"] not in methods:
            continue
        if keep_q is not None and g["query"] not in keep_q:
            continue
        key = (str(g.get("dataset", args.dataset)), str(g.get("setting", args.setting)), g["query"], g["method"])
        if key in done:
            continue
        todo.append(g)
        if args.limit and len(todo) >= args.limit:
            break
    todo.sort(key=lambda g: -(len(g["members"]) if g.get("members") is not None else 0))
    inst = Instance(args.instance, mmap=True)
    shared = {"inst": inst, "cfg": cfg, "variants": variants, "dataset": args.dataset, "setting": args.setting,
              "pos": np.full(inst.n, -1, dtype=np.int32),
              "obs": ObservedWeights(args.weights, inst.n) if args.weights else None}
    if any(v["kind"] == "observed" for v in variants) and shared["obs"] is None:
        raise SystemExit("observed-weight variants need --weights")
    write_json(os.path.join(args.out, "manifest.json"), {
        "version": VERSION, "instance": os.path.abspath(args.instance), "groups": os.path.abspath(args.groups),
        "weights": os.path.abspath(args.weights) if args.weights else None, "lcc_only": bool(args.lcc_only),
        "queries_from": os.path.abspath(args.queries_from) if args.queries_from else None,
        "n_groups_todo": len(todo), "n_groups_done_before": len(done), "config": cfg, "kit_sha256": kit_digest(),
        "host": host_info(), "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    wr = JsonlWriter(rows_path, append=True)
    wg = JsonlWriter(grp_path, append=True)

    def emit(res):
        rows, grow = res
        wr.write_many(rows)
        wg.write_many([grow if grow is not None else {k: rows[0][k] for k in ("dataset", "setting", "query", "k",
                                                                               "method", "size", "skipped")}])

    t0 = time.time()
    run_pool(shared, _work, todo, args.workers, log_every=200, label="e1x", on_result=emit)
    wr.close()
    wg.close()
    Path(args.out, "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[e1x] {args.out}: {len(todo)} groups in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
