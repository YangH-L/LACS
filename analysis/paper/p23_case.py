"""P2-3: E6 case figure (plan docs/PLANS.md (lacs_p2_plan_20261026) section 3).

Selection (fixed in the plan before any P2 result):
  1. among the queries with a non-empty LCC (k = 3, theta_q50), keep those whose sac_sized group has at
     least one member facing another member beyond its own acceptance region (eccentricity > theta_u,
     the Q2 definition of e6/static.py: instance positions and tolerances, relative tolerance 1e-9);
  2. pick the query whose LCC size is closest to the median LCC size over all non-empty-LCC queries;
     ties -> the smallest query id.
Figure: three stacked panels (LCC exact, sac_sized, kr_global), x = DW-NOMINATE dimension 1, one row per
member: dot coloured by caucus, line = own acceptance interval [x - theta, x + theta] (public theta_q50,
uninflated), hollow dot = member facing someone beyond its own acceptance region, the querying member
marked by a ring.
Usage: python3 -I p23_case.py <runs/e6 dir> <out dir>
"""
import gzip, json, sys
from pathlib import Path
import numpy as np, pandas as pd

E6, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RTOL = 1e-9
X = np.load(E6 / "inst/X_default.npy").astype(np.float64)
TH = np.load(E6 / "inst/theta_default.npy").astype(np.float64)
users = pd.read_csv(E6 / "geo/users.csv")
pub = pd.read_csv(E6 / "public/e6_members.csv")
pub = pub[pub["eligible"]].set_index("idx").sort_index()
assert (users["idx"].to_numpy() == np.arange(len(users))).all() and (pub.index.to_numpy() == users["idx"].to_numpy()).all()
assert (pub["bioguide"].to_numpy() == users["user"].to_numpy()).all()
G = {}
with gzip.open(E6 / "e1_input/groups.jsonl.gz", "rt") as fh:
    for line in fh:
        r = json.loads(line)
        G[(int(r["query"]), r["method"])] = np.asarray(sorted(r["members"] or []), np.int64)


def ecc_viol(H):
    if len(H) < 2:
        return np.zeros(len(H), bool)
    P = X[H]
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    return D.max(1) > TH[H] * (1 + RTOL)


queries = sorted({q for q, m in G if m == "lcc_exact"})
lcc_q = [q for q in queries if len(G[(q, "lcc_exact")]) >= 2]
lcc_size = pd.Series({q: len(G[(q, "lcc_exact")]) for q in lcc_q})
med = float(lcc_size.median())
eligible = [q for q in lcc_q if ecc_viol(G.get((q, "sac_sized"), np.empty(0, np.int64))).any()]
dist = (lcc_size.loc[eligible] - med).abs()
best = dist.min()
q_sel = int(min(q for q in eligible if dist.loc[q] == best))
sel = {"queries": len(queries), "lcc_nonempty": len(lcc_q), "median_lcc_size": med,
       "rule1_eligible": len(eligible), "rule1_share": len(eligible) / len(lcc_q),
       "closest_distance": float(best), "ties": sorted(int(q) for q in eligible if dist.loc[q] == best),
       "query": q_sel, "lcc_size": int(lcc_size.loc[q_sel])}
panels = []
for m, label in (("lcc_exact", "LCC (exact)"), ("sac_sized", "Spatial community, size-matched"),
                 ("kr_global", "(k,r)-core, r = median tolerance")):
    H = G.get((q_sel, m), np.empty(0, np.int64))
    v = ecc_viol(H)
    P = X[H]
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)) if len(H) else np.zeros((0, 0))
    lim = np.minimum(TH[H][:, None], TH[H][None, :]) * (1 + RTOL)
    iu = np.triu_indices(len(H), 1)
    panels.append({"method": m, "label": label, "size": int(len(H)), "members_beyond": int(v.sum()),
                   "pairs": int(len(iu[0])), "pairs_rejected": int((D[iu] > lim[iu]).sum()) if len(H) else 0,
                   "diam": float(D.max()) if len(H) else None, "min_theta": float(TH[H].min()) if len(H) else None,
                   "iii": bool(D.max() <= TH[H].min() * (1 + RTOL)) if len(H) >= 2 else None,
                   "members": H.tolist(), "viol": v.tolist()})
sel["panels"] = [{k: p[k] for k in p if k not in ("members", "viol")} for p in panels]
q_row = pub.loc[q_sel]
sel["query_member"] = {"caucus": q_row["caucus"], "chamber": q_row["chamber"], "dim1": float(q_row["dim1"]),
                       "theta_q50_1d": float(q_row["theta_q50_1d"])}
(OUT / "p23_selection.json").write_text(json.dumps(sel, indent=1))
print(json.dumps(sel, indent=1))

# ---------------------------------------------------------------- figure --
# Layout: the plan (section 3.2) asked for three stacked panels; the paper's space for the figure (at most 2.4 in tall) leaves too little height for 187 member rows
# stacked, so the three panels sit side by side (same content, same encoding). --stacked restores the plan's
# layout for the technical report.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

STACKED = "--stacked" in sys.argv
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "legend.fontsize": 6.2, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.0})
COL = {"D": "#2a78d6", "R": "#e34948"}
BAND, INK2 = "#e6e5e1", "#52514e"
TITLES = {"lcc_exact": "(a) LCC", "sac_sized": "(b) Size-matched SAC", "kr_global": "(c) Global $(k,r)$-core"}
x1 = pub["dim1"].to_numpy(float)
th1 = pub["theta_q50_1d"].to_numpy(float)
xmin = min((x1[p["members"]] - th1[p["members"]]).min() for p in panels if p["size"]) - 0.02
xmax = max((x1[p["members"]] + th1[p["members"]]).max() for p in panels if p["size"]) + 0.02


def draw(ax, p, ms=2.6, lw=0.45):
    H = np.asarray(p["members"], np.int64)
    v = np.asarray(p["viol"], bool)
    o = np.lexsort((H, x1[H]))
    H, v = H[o], v[o]
    if len(H):
        ax.axvspan(x1[H].min(), x1[H].max(), color=BAND, lw=0, zorder=0)
    for yi, hh, vi in zip(range(len(H)), H, v):
        c = COL[pub.loc[hh, "caucus"]]
        ax.plot([x1[hh] - th1[hh], x1[hh] + th1[hh]], [yi, yi], color=c, lw=lw, alpha=0.95 if vi else 0.45,
                solid_capstyle="butt", zorder=1)
        if vi:
            ax.scatter([x1[hh]], [yi], s=ms * 3.2, facecolor="white", edgecolor=c, linewidth=0.7, zorder=3)
        else:
            ax.scatter([x1[hh]], [yi], s=ms, facecolor=c, edgecolor="none", zorder=2)
        if hh == q_sel:
            ax.scatter([x1[hh]], [yi], s=ms * 9, facecolor="none", edgecolor="black", linewidth=0.55, zorder=4)
    ax.set_ylim(-1.5, max(len(H), 4) + 0.5)
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


present = {pub.loc[int(h), "caucus"] for p in panels for h in p["members"]}
handles = [Line2D([], [], marker="o", ls="", markersize=2.4, markerfacecolor=COL[c], markeredgecolor="none", label=lab)
           for c, lab in (("D", "Democratic caucus"), ("R", "Republican")) if c in present]
handles += [Line2D([], [], color=INK2, lw=0.6, label=r"own interval $[x-\theta, x+\theta]$"),
            Line2D([], [], marker="o", ls="", markersize=3.6, markerfacecolor="white", markeredgecolor=INK2,
                   label="faces a member beyond it"),
            Patch(facecolor=BAND, edgecolor="none", label="span of the group"),
            Line2D([], [], marker="o", ls="", markersize=4.8, markerfacecolor="none", markeredgecolor="black",
                   markeredgewidth=0.55, label="querying member")]
if STACKED:
    ROW_IN, PAD_IN, LEG_IN, AX_IN = 0.0125, 0.23, 0.42, 0.30
    sizes = [max(p["size"], 4) for p in panels]
    H_IN = LEG_IN + AX_IN + sum(s * ROW_IN + PAD_IN for s in sizes)
    fig = plt.figure(figsize=(3.5, H_IN))
    top = H_IN - LEG_IN
    axes = []
    for p, s in zip(panels, sizes):
        h = s * ROW_IN
        ax = fig.add_axes([0.03, (top - PAD_IN - h) / H_IN, 0.95, h / H_IN])
        top -= PAD_IN + h
        draw(ax, p, ms=3.5, lw=0.5)
        if p is not panels[-1]:
            ax.tick_params(axis="x", labelbottom=False)
        n_b = p["members_beyond"]
        tail = "all members accept each other" if n_b == 0 else f"{n_b} face a member beyond their tolerance"
        ax.set_title(f"{TITLES[p['method']]}: {p['size']} members, {tail}", loc="left", pad=1.5, x=0.0)
        axes.append(ax)
    axes[-1].set_xlabel("DW-NOMINATE dimension 1", labelpad=1.5)
    fig.legend(handles=handles, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.01, 1.0),
               handletextpad=0.4, columnspacing=0.9, borderaxespad=0.15, handlelength=1.4)
    name = "e6_case_stacked"
else:
    W, H_IN = 3.5, 2.3  # the b draft reserves a 2.3 in box for this figure
    fig = plt.figure(figsize=(W, H_IN))
    left, right, gap = 0.012, 0.995, 0.022
    pw = (right - left - 2 * gap) / 3
    bottom, top_ax = 0.125, 0.79
    axes = []
    for i, p in enumerate(panels):
        ax = fig.add_axes([left + i * (pw + gap), bottom, pw, top_ax - bottom])
        draw(ax, p)
        n_b = p["members_beyond"]
        ax.set_title(f"{TITLES[p['method']]}\n{p['size']} members" + (f", {n_b} hollow" if n_b else ", none hollow"),
                     loc="left", pad=1.5, x=0.0, fontsize=6.4, linespacing=1.05)
        ax.set_xticks([0.3, 0.5, 0.7, 0.9])
        axes.append(ax)
    fig.text(0.5, 0.012, "DW-NOMINATE dimension 1", ha="center", va="bottom", fontsize=7)
    n_c = len(present)  # caucus entries first in `handles`
    cauc, (own, hollow, span, qm) = handles[:n_c], handles[n_c:]
    # legend fills column by column; this order reads row-wise as: caucus | own interval | hollow, then span | query
    order = cauc + [span, own, qm, hollow] if n_c == 1 else cauc + [own, hollow, span, qm]
    fig.legend(handles=order, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005),
               handletextpad=0.3, columnspacing=0.7, borderaxespad=0.1, handlelength=1.2, fontsize=5.9)
    name = "e6_case"
fig.savefig(OUT / f"{name}.pdf")
fig.savefig(OUT / f"{name}.png", dpi=250)
print("figure:", OUT / f"{name}.pdf", fig.get_size_inches())
