"""Summary of an e1x.open_sweep run (or several), with the theorem check.

  python -m e1x.summarize --runs runs/e1_open/gowalla runs/e1_open/brightkite --out runs/e1_open/report \
      [--p0-rows gowalla=runs_p0/gowalla/e1/default/e1_rows.jsonl.gz ...]

Writes e1x_summary.csv (columns dataset, method, variant, w, groups, share_broken, ci_lo, ci_hi,
share_cond_special, share_cond_refined, theorem_violations, ...), e1x_first_w.csv (per method, how
many groups first lose an initially accepted interaction at each open weight w, or never),
e1x_check.json and E1X_SUMMARY.md. Theorem check (must be 0, otherwise the numbers are not to be
interpreted): a group with (iii) never breaks under the closed convex variants (hk, follow, dw), and
a group meeting the refined open condition (a)+(b) never breaks under an open variant.
With --p0-rows, the closed hk_eq_G and the open w = 1 rows are compared group by group with P0's
hk_eq_G and hk_open_G (same groups, same outside-neighbour samples): they should agree exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "p0", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from p0kit.formats import read_jsonl  # noqa: E402

CONVEX_CLOSED = ("hk", "follow", "dw")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def load_run(d: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = pd.DataFrame(list(read_jsonl(str(d / "e1x_rows.jsonl.gz"))))
    grps = pd.DataFrame(list(read_jsonl(str(d / "e1x_groups.jsonl.gz"))))
    man = json.loads((d / "manifest.json").read_text())
    return rows, grps, man


def summarize(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out, firsts, checks = [], [], {"violations": [], "runs": []}
    for d in runs:
        rows, grps, man = load_run(d)
        cfg_vars = {v["name"]: v for v in man["config"]["variants"]}
        grid = sorted({0.0} | {float(v["open_w"]) for v in cfg_vars.values() if "open_w" in v})
        skipped = {}
        if "skipped" in grps.columns:
            sk = grps["skipped"].fillna("")
            skipped = {k: int(v) for k, v in sk[sk != ""].str.split(":").str[0].value_counts().items()}
        if "skipped" in rows.columns:
            rows = rows[rows["skipped"].fillna("") == ""]
        if "skipped" in grps.columns:
            grps = grps[grps["skipped"].fillna("") == ""]
        checks["runs"].append({"dir": str(d), "groups": int(len(grps)), "rows": int(len(rows)),
                               "groups_skipped": skipped, "kit_sha256": man.get("kit_sha256")})
        checks["errors_total"] = checks.get("errors_total", 0) + int(skipped.get("error", 0))
        gk = ["dataset", "setting", "query", "method"]
        st = grps.set_index(gk)
        for (ds, meth), gm in grps.groupby(["dataset", "method"]):
            n_g = len(gm)
            sp = float((gm["cond_special"] == 1).mean())
            rf = float((gm["cond_refined"] == 1).mean())
            fw = gm["first_open_w_broken"]
            dist = {"never": int(fw.isna().sum())}
            for w in grid:
                dist[f"w={w:g}"] = int((fw == w).sum())
            firsts.append({"dataset": ds, "method": meth, "groups": n_g, **dist})
            rm = rows[(rows["dataset"] == ds) & (rows["method"] == meth)]
            for var, rv in rm.groupby("variant", sort=False):
                spec = cfg_vars.get(var, {})
                k = int(rv["broken_any"].sum())
                n = int(len(rv))
                lo, hi = wilson(k, n)
                joined = rv.join(st[["sat_iii", "cond_refined"]], on=gk)
                is_open = ("open_w" in spec) or bool(spec.get("open"))
                if is_open:
                    bad = joined[(joined["cond_refined"] == 1) & joined["broken_any"]]
                elif spec.get("observed") or spec.get("model", "hk") in CONVEX_CLOSED:
                    bad = joined[joined["sat_iii"].astype(bool) & joined["broken_any"]]
                else:  # sjt: Theorem 2 needs LCC^rej; not checked here
                    bad = joined.iloc[0:0]
                for _, b in bad.iterrows():
                    checks["violations"].append({"dataset": ds, "method": meth, "variant": var,
                                                 "query": int(b["query"])})
                out.append({"dataset": ds, "method": meth, "variant": var,
                            "w": spec.get("open_w", ("observed" if spec.get("observed") else 0.0)),
                            "open": is_open, "groups": n, "share_broken": k / n if n else math.nan,
                            "ci_lo": lo, "ci_hi": hi, "share_cond_special": sp, "share_cond_refined": rf,
                            "share_cond_refined_boundary": float((gm["cond_refined"] == -1).mean()),
                            "share_sat_iii": float(gm["sat_iii"].astype(bool).mean()),
                            "median_size": float(gm["size"].median()),
                            "mean_largest_cluster": float(rv["mean_largest_frac"].mean()),
                            "share_consensus": float(rv["p_consensus"].mean()),
                            "theorem_violations": int(len(bad))})
    checks["theorem_violations_total"] = len(checks["violations"])
    return pd.DataFrame(out), pd.DataFrame(firsts), checks


def compare_p0(run: Path, p0_rows: Path, mapping=(("hk_eq_G", "hk_eq_G"), ("open_w1", "hk_open_G"))) -> dict:
    rows, _, _ = load_run(run)
    p0 = pd.DataFrame([r for r in read_jsonl(str(p0_rows)) if r.get("variant") in {b for _, b in mapping}])
    res = {}
    key = ["query", "method"]
    for mine, theirs in mapping:
        a = rows[rows["variant"] == mine].set_index(key)
        b = p0[p0["variant"] == theirs].set_index(key)
        common = a.index.intersection(b.index)
        if not len(common):
            res[f"{mine}~{theirs}"] = {"common": 0}
            continue
        same_b = (a.loc[common, "p_broken"].to_numpy() == b.loc[common, "p_broken"].to_numpy())
        same_c = np.isclose(a.loc[common, "mean_largest_frac"].to_numpy(float),
                            b.loc[common, "mean_largest_frac"].to_numpy(float))
        res[f"{mine}~{theirs}"] = {"common": int(len(common)), "same_p_broken": int(same_b.sum()),
                                   "same_largest_cluster": int(same_c.sum())}
    return res


def _md(df: pd.DataFrame) -> str:
    def f(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + " --- |" * len(df.columns)]
    lines += ["| " + " | ".join(f(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--p0-rows", nargs="*", default=[], help="dataset=path of P0's e1_rows.jsonl.gz")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    S, Fw, C = summarize(args.runs)
    for spec in args.p0_rows:
        ds, path = spec.split("=", 1)
        run = next((r for r in args.runs if r.name == ds), None)
        if run is not None and Path(path).exists():
            C.setdefault("p0_agreement", {})[ds] = compare_p0(run, Path(path))
    S.to_csv(args.out / "e1x_summary.csv", index=False)
    Fw.to_csv(args.out / "e1x_first_w.csv", index=False)
    (args.out / "e1x_check.json").write_text(json.dumps(C, indent=1))
    L = ["# E1x summary", "", f"theorem violations: {C['theorem_violations_total']} (must be 0); groups that "
         f"failed with an error: {C.get('errors_total', 0)} (must be 0)", "",
         "Skipped groups per run: " + json.dumps([r["groups_skipped"] for r in C["runs"]]), ""]
    if C.get("p0_agreement"):
        L += ["P0 agreement: " + json.dumps(C["p0_agreement"]), ""]
    if len(S):
        L += [_md(S[["dataset", "method", "variant", "w", "groups", "share_broken", "ci_lo", "ci_hi",
                     "share_cond_special", "share_cond_refined", "theorem_violations"]]), ""]
    if len(Fw):
        L += ["## First open weight with a broken interaction", "", _md(Fw.fillna(0)), ""]
    (args.out / "E1X_SUMMARY.md").write_text("\n".join(L))
    print(f"[e1x] summary: {len(S)} rows, theorem violations {C['theorem_violations_total']}, "
          f"errors {C.get('errors_total', 0)}")
    return 1 if (C["theorem_violations_total"] or C.get("errors_total", 0)) else 0


if __name__ == "__main__":
    sys.exit(main())
