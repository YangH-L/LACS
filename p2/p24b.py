"""P2-4b summary (plan section 4.4): LCC groups found with s = 0.8 and s = 1, equal-weight HK on G[H] whose
tolerances are drawn around the estimate (P0's noise 0.3, 10 draws per group).

  PYTHONPATH=p0:. python -m p2.p24b --root runs_p2/p24b --out runs_p2/p24b/report

Input: runs_p2/p24b/<dataset>/s<s>/dyn/e1x_rows.jsonl.gz (e1x.open_sweep with p2/configs/p24b_dynamics.yaml).
Per dataset, s and variant: groups, share of groups that lose an initially accepted interaction in at least one
draw (P0's "share_breakable"), with a Wilson 95% interval, and the mean share of draws with a loss. Also the same
on the queries that have an LCC at both s. Theorem check: without noise (hk_eq_G) no LCC group may lose an
interaction (exit code 1 otherwise).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "p0", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def load(root: Path) -> pd.DataFrame:
    from p0kit.formats import read_jsonl

    frames = []
    for f in sorted(root.glob("*/s*/dyn/e1x_rows.jsonl.gz")):
        ds, s = f.parts[-4], f.parts[-3]
        d = pd.DataFrame(list(read_jsonl(str(f))))
        d["ds"], d["s"] = ds, float(s[1:])
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m p2.p24b", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.root))
    if df.empty:
        print("!! no P2-4b rows")
        return 1
    skipped = df[df.get("skipped").notna()] if "skipped" in df.columns else df.iloc[0:0]
    ok = df[df["skipped"].isna()] if "skipped" in df.columns else df
    ok = ok[ok["method"] == "lcc_exact"]
    rows, viol = [], 0
    for (ds, var), g in ok.groupby(["ds", "variant"]):
        qs = {s: set(h["query"]) for s, h in g.groupby("s")}
        common = set.intersection(*qs.values()) if len(qs) > 1 else set()
        for s, h in g.groupby("s"):
            for scope, hh in (("all", h), ("common", h[h["query"].isin(common)])):
                n, k = len(hh), int(hh["broken_any"].astype(bool).sum())
                lo, hi = wilson(k, n)
                rows.append({"dataset": ds, "variant": var, "s": s, "scope": scope, "groups": n, "broken": k,
                             "share_broken": k / n if n else math.nan, "ci_lo": lo, "ci_hi": hi,
                             "mean_share_of_draws_broken": float(hh["p_broken"].mean()) if n else math.nan,
                             "median_size": float(hh["size"].median()) if n else math.nan})
            if var == "hk_eq_G":
                viol += int(h["broken_any"].astype(bool).sum())
    S = pd.DataFrame(rows).sort_values(["dataset", "variant", "scope", "s"])
    S.to_csv(out / "p24b_summary.csv", index=False)
    sk = skipped.groupby(["ds", "s", "variant", "skipped"]).size().reset_index(name="n") if len(skipped) else None
    lines = ["# P2-4b: LCC groups at s = 0.8 and s = 1 under tolerance noise", "",
             f"theorem check (hk_eq_G, no noise): {viol} LCC groups lost an interaction (must be 0)", "",
             "| " + " | ".join(S.columns) + " |", "|" + " --- |" * len(S.columns)]
    for r in S.itertuples(index=False):
        lines.append("| " + " | ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in r) + " |")
    if sk is not None:
        lines += ["", "skipped (group, variant) pairs:", sk.to_string(index=False)]
    (out / "P24B_SUMMARY.md").write_text("\n".join(lines) + "\n")
    (out / "p24b_check.json").write_text(json.dumps({"theorem_violations": viol}, indent=1))
    print("\n".join(lines))
    return 0 if viol == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
