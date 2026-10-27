"""Figure version of the meetup comparison (LCC minus size-matched SAC), Section 6.4.

Reads meetup_values.json, which make_tables.py writes from the result bundles together with
tab_meetup.tex (the same numbers; the table appears only in the technical report), and writes
figures/fig_meetup.pdf for one IEEE column.

Usage: python3 -I make_fig_meetup.py <paper_dir>
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PAPER = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
vals = json.load(open(os.path.join(HERE, "meetup_values.json")))["datasets"]

# Times-compatible serif, embedded as TrueType (IEEE PDF eXpress needs embedded fonts).
for f in ["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
          "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 7, "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "xtick.major.size": 2,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
})

DATASETS = ["Gowalla", "Brightkite", "Yelp"]
ESTS = [("raw", "raw"), ("a", "(a)"), ("b", "(b)")]
STYLE = {"raw": dict(color="0.55", marker="o", ms=2.8),
         "a": dict(color="black", marker="o", ms=2.8),
         "b": dict(color="black", marker="D", ms=2.4)}

rows, y = [], 0.0
for ds in DATASETS:
    for key, label in ESTS:
        rows.append((ds, key, label, y))
        y -= 1.0
    y -= 0.45

fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.36), sharey=True,
                         gridspec_kw=dict(wspace=0.07, left=0.2, right=0.995, top=0.895, bottom=0.215))
panels = [(axes[0], "pairs", "Member pairs that co-visit", (-0.62, 1.85), [-0.5, 0, 0.5, 1, 1.5]),
          (axes[1], "members", "Members who co-visit anyone", (-2.3, 5.1), [-2, 0, 2, 4])]
for ax, metric, title, xlim, xticks in panels:
    ax.axvline(0, color="0.6", lw=0.5, ls=(0, (2, 1.5)), zorder=0)
    for ds, key, label, yy in rows:
        v = vals[ds][metric].get(key)
        if v is None:
            continue
        st = STYLE[key]
        ax.errorbar([v["est"]], [yy], xerr=[[v["est"] - v["lo"]], [v["hi"] - v["est"]]], fmt=st["marker"],
                    ms=st["ms"], color=st["color"], mfc=st["color"], mec=st["color"], elinewidth=0.8,
                    capsize=1.6, capthick=0.6)
    ax.set_title(title, fontsize=7, pad=2.5)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([("$-$" if t < 0 else "") + f"{abs(t):g}" for t in xticks])
    ax.tick_params(axis="x", pad=1.5)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
axes[0].set_yticks([r[3] for r in rows])
axes[0].set_yticklabels([r[2] for r in rows])
axes[0].tick_params(axis="y", length=0, pad=2)
axes[1].tick_params(axis="y", length=0)
axes[0].set_ylim(rows[-1][3] - 0.6, rows[0][3] + 0.6)
for ds in DATASETS:
    ymid = [r[3] for r in rows if r[0] == ds][1]
    axes[0].annotate(ds, xy=(0, ymid), xycoords=("axes fraction", "data"), xytext=(-17, 0),
                     textcoords="offset points", ha="right", va="center", fontsize=7, fontweight="bold",
                     annotation_clip=False)
fig.supxlabel("LCC minus size-matched SAC (percentage points)", fontsize=6.5, y=0.015, x=0.6)
out = os.path.join(PAPER, "figures", "fig_meetup.pdf")
fig.savefig(out)
fig.savefig(os.path.join(HERE, "fig_meetup_preview.png"), dpi=300)
print("wrote", out)
