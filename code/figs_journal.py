"""Journal-style figures for the RAVI paper: no in-figure titles or footnotes (captions carry them) and no
value annotations inside the plots (all values are given in the tables). Same data and encodings as figs_ravi.py.
Usage: python figs_journal.py results_dir out_dir
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

RES, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
MODELS = [("rave", "RAVI", "#2a78d6"), ("rave_noanchor", "RAVI w/o anchor", "#eb6834"),
          ("lam", "LAM", "#1baf7a"), ("signrev", "Sign-rev. LAM", "#eda100")]
ORACLE = ("oracle", "Two-direction ref.", "#888780")         # trained on both directions; not an upper bound
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
DIRS = [("I", "I→R"), ("R", "R→I")]
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#c2491f", "#f3b39a", "#f0efec", "#86b6ef", "#184f95"])
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})

runs = defaultdict(list)
for f in glob.glob(os.path.join(RES, "[IR]_*.json")):
    r = json.load(open(f))
    if r["model"] in ("lam", "signrev", "rave_noanchor", "rave", "oracle"):
        runs[(r["train"], r["model"])].append(r)                                  # all seeds (5 folds x 3 seeds)


def vals(d, m, key):
    out = []
    for r in sorted(runs[(d, m)], key=lambda r: (r["fold"], r.get("seed", 0))):
        v = (r.get("fewshot_fwd_rel_err", {}).get(key.split(":")[1]) if key.startswith("fewshot:") else r.get(key))
        if v is not None:
            out.append(v)
    return np.array(out, float)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


# 1. key metrics (small multiples)
PANELS = [("reversibility_auroc", "AUROC (↑)", 0.5), ("end_dist_ratio", "end-distance ratio (↓)", 1.0),
          ("fewshot:5", "opposite-task error, k=5 (↓)", 1.0), ("action_cos", "action alignment (↑)", None)]
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.0)); w = 0.19
for ax, (key, ylab, ref) in zip(axes, PANELS):
    for gi, (d, _) in enumerate(DIRS):
        for mi, (m, _, col) in enumerate(MODELS):
            v = vals(d, m, key); x = gi + (mi - 1.5) * (w + 0.012)
            ax.bar(x, v.mean(), w, color=col, edgecolor=SURF, linewidth=1, zorder=2)
            ax.errorbar(x, v.mean(), v.std(), color=INK2, lw=0.8, capsize=1.8, zorder=3)
            ax.scatter(np.full(len(v), x), v, s=5, color=INK, alpha=0.35, zorder=4, linewidths=0)
        if key != "fewshot:5":
            o = vals(d, ORACLE[0], key)
            ax.plot([gi - 0.44, gi + 0.44], [o.mean()] * 2, ls="--", lw=1, color=ORACLE[2], zorder=5)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=0.8, ls=":", zorder=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels([n for _, n in DIRS]); ax.set_ylabel(ylab)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True); ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, max(ax.get_ylim()[1], (ref or 0) * 1.08) * 1.05)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MODELS] + [plt.Line2D([], [], ls="--", color=ORACLE[2])]
fig.legend(handles, [n for _, n, _ in MODELS] + [ORACLE[1]], loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.06), fontsize=8.5)
fig.tight_layout(w_pad=2.2); save(fig, "fig_compare_bars")

# 2. scorecard heatmap (shading only; values in Table 1 and the appendix)
METRICS = [("forward_rel_err", "fwd. err.", -1), ("end_dist_ratio", "end dist.", -1), ("action_cos", "act. align.", 1),
           ("reversibility_auroc", "AUROC", 1), ("fewshot:5", "few-shot", -1), ("joint_probe_nmse", "joint probe", -1)]
rows = MODELS + [ORACLE]; cols = [(d, k, l, s) for d, _ in DIRS for k, l, s in METRICS]
raw = np.full((len(rows), len(cols)), np.nan); score = raw.copy()
for j, (d, k, _, s) in enumerate(cols):
    for i, (m, _, _) in enumerate(rows):
        v = vals(d, m, k); raw[i, j] = v.mean() if len(v) else np.nan
    base = raw[: len(MODELS), j] * s; lo, hi = np.nanmin(base), np.nanmax(base)
    score[:, j] = np.clip((raw[:, j] * s - lo) / (hi - lo + 1e-12), 0, 1)
fig, ax = plt.subplots(figsize=(11.0, 2.9))
im = ax.imshow(np.ma.masked_invalid(np.where(np.isnan(raw), np.nan, score)), cmap=BLUE, vmin=0, vmax=1, aspect="auto")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for _, n, _ in rows])
ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{l}\n{dn}" for d, _, l, _ in cols for dn in [dict(DIRS)[d]]], fontsize=8)
ax.axvline(len(METRICS) - 0.5, color=SURF, lw=4); ax.axhline(len(MODELS) - 0.5, color=SURF, lw=4)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.outline.set_visible(False)
cb.set_label("relative score", fontsize=8, color=INK2)
save(fig, "fig_scorecard_heatmap")

# 3. few-shot curves
ks = [0, 1, 3, 5]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.9))
for ax, (d, dname) in zip(axes, DIRS):
    for m, name, col in MODELS:
        Y = np.stack([vals(d, m, f"fewshot:{k}") for k in ks], 1); mu, sd = Y.mean(0), Y.std(0)
        ax.fill_between(ks, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
        ax.plot(ks, mu, color=col, lw=2, marker="o", ms=4.5, mec=SURF, mew=1.2, label=name)
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(ks); ax.set_xlabel(f"adaptation episodes k ({dname})")
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("opposite-task error (↓)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(); save(fig, "fig_fewshot_curves")

# 4. per-fold AUROC heatmap (colour only; values in the appendix tables)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.7))
norm = TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=0.85)
for ax, (d, dname) in zip(axes, DIRS):
    M = np.full((len(rows), 5), np.nan)
    for i, (m, _, _) in enumerate(rows):
        for f in range(5):                                                       # mean over seeds per fold
            v = [r["reversibility_auroc"] for r in runs[(d, m)] if r["fold"] == f]
            M[i, f] = np.mean(v) if v else np.nan
    im = ax.imshow(M, cmap=DIVERGE, norm=norm, aspect="auto")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{k}" for k in range(5)]); ax.set_xlabel(f"fold ({dname})")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for _, n, _ in rows] if d == "I" else [])
    ax.axhline(len(MODELS) - 0.5, color=SURF, lw=4)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01); cb.outline.set_visible(False)
cb.set_label("AUROC", fontsize=8, color=INK2)
save(fig, "fig_auroc_folds_heatmap")
print("saved", sorted(os.listdir(OUT)))
