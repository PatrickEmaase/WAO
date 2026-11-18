"""Comparative figures from results/*.json (fold level): bars, scorecard heatmap, few-shot curves, per-fold AUROC heatmap."""
import glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results") if glob.glob(os.path.join(ROOT, "results", "[IR]_*.json")) \
    else os.path.join(ROOT, "..", "results")
OUT = os.path.join(RES, "figures_ravi"); os.makedirs(OUT, exist_ok=True)

# categorical slots in fixed order (validated: adjacent CVD dE >= 9.1, normal-vision >= 22.9); oracle = neutral reference
MODELS = [("rave", "RAVI", "#2a78d6"), ("rave_noanchor", "RAVI w/o anchor", "#eb6834"),
          ("lam", "LAM", "#1baf7a"), ("signrev", "Sign-rev. LAM", "#eda100")]
ORACLE = ("oracle", "Two-direction reference", "#888780")     # trained on both directions; not an upper bound
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
DIRS = [("I", "Insertion → removal"), ("R", "Removal → insertion")]
BLUE = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#c2491f", "#f3b39a", "#f0efec", "#86b6ef", "#184f95"])

plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})

runs = defaultdict(list)
for f in glob.glob(os.path.join(RES, "[IR]_*.json")):
    r = json.load(open(f))
    if r.get("seed", 0) == 0:                                                    # figures show seed 0 per fold
        runs[(r["train"], r["model"])].append(r)


def vals(d, m, key):
    out = []
    for r in sorted(runs[(d, m)], key=lambda r: r["fold"]):
        v = (r.get("fewshot_fwd_rel_err", {}).get(key.split(":")[1]) if key.startswith("fewshot:")
             else r.get(key))                                                  # oracle has no few-shot entry
        if v is not None:
            out.append(v)
    return np.array(out, float)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. comparison bars (small multiples)
PANELS = [("reversibility_auroc", "Reversibility AUROC ↑", 0.5, "chance"),
          ("end_dist_ratio", "End-distance ratio ↓", 1.0, "no progress"),
          ("fewshot:5", "Opposite-task error, k = 5 ↓", 1.0, "predict no change"),
          ("action_cos", "Action alignment ↑", None, None)]
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4))
w = 0.19
for ax, (key, title, ref, reflabel) in zip(axes, PANELS):
    for gi, (d, dname) in enumerate(DIRS):
        for mi, (m, name, col) in enumerate(MODELS):
            v = vals(d, m, key); x = gi + (mi - 1.5) * (w + 0.012)
            ax.bar(x, v.mean(), w, color=col, edgecolor=SURF, linewidth=1, zorder=2)
            ax.errorbar(x, v.mean(), v.std(), color=INK2, lw=0.8, capsize=1.8, zorder=3)
            ax.scatter(np.full(len(v), x), v, s=5, color=INK, alpha=0.35, zorder=4, linewidths=0)
            ax.text(x, 0.02, f"{v.mean():.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90, zorder=6,
                    color="#ffffff" if mi < 2 else INK, transform=ax.get_xaxis_transform())   # inside the bar base
        if key != "fewshot:5":
            o = vals(d, ORACLE[0], key)
            ax.plot([gi - 0.44, gi + 0.44], [o.mean()] * 2, ls="--", lw=1, color=ORACLE[2], zorder=5)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=0.8, ls=":", zorder=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["I → R", "R → I"])
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_xlim(-0.55, 1.55)
    top = max(ax.get_ylim()[1], (ref or 0) * 1.08); ax.set_ylim(0, top * 1.12)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in MODELS] + [plt.Line2D([], [], ls="--", color=ORACLE[2])]
fig.legend(handles, [n for _, n, _ in MODELS] + [ORACLE[1]], loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 1.08), fontsize=8.5)
fig.suptitle("Koopman models and the plain LAM stay stable; sign reversal does not",
             x=0.01, ha="left", y=1.17, fontsize=11, color=INK)
fig.text(0.01, -0.06, "Bars: mean over 5 folds; whiskers: ±1 std; dots: folds. Dotted lines: chance (AUROC 0.5), "
         "no progress toward the target (ratio 1.0), predicting no change (error 1.0).\n"
         "I → R = trained on insertion, tested on removal; R → I the reverse.", fontsize=7.5, color=INK2)
fig.tight_layout(w_pad=2.2)
save(fig, "fig_compare_bars")

# ---------------------------------------------------------------- 2. scorecard heatmap
METRICS = [("forward_rel_err", "Forward err.", -1), ("end_dist_ratio", "End dist.", -1),
           ("action_cos", "Action align.", 1), ("reversibility_auroc", "Rev. AUROC", 1),
           ("fewshot:5", "Opp. task k=5", -1), ("joint_probe_nmse", "Joint probe", -1)]
rows = MODELS + [ORACLE]
cols = [(d, k, lbl, s) for d, _ in DIRS for k, lbl, s in METRICS]
raw = np.full((len(rows), len(cols)), np.nan); score = raw.copy()
for j, (d, k, _, s) in enumerate(cols):
    for i, (m, _, _) in enumerate(rows):
        v = vals(d, m, k); raw[i, j] = v.mean() if len(v) else np.nan
    base = raw[: len(MODELS), j] * s
    lo, hi = np.nanmin(base), np.nanmax(base)
    score[:, j] = np.clip((raw[:, j] * s - lo) / (hi - lo + 1e-12), 0, 1)
fig, ax = plt.subplots(figsize=(11.5, 3.3))
im = ax.imshow(score, cmap=BLUE, vmin=0, vmax=1, aspect="auto")
for i in range(len(rows)):
    for j in range(len(cols)):
        if np.isnan(raw[i, j]):
            ax.text(j, i, "—", ha="center", va="center", color=INK2, fontsize=8); continue
        ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center", fontsize=7.6,
                color="#ffffff" if score[i, j] > 0.55 else INK)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for _, n, _ in rows])
ax.set_xticks(range(len(cols))); ax.set_xticklabels([lbl for _, _, lbl, _ in cols], rotation=35, ha="right")
ax.axvline(len(METRICS) - 0.5, color=SURF, lw=4); ax.axhline(len(MODELS) - 0.5, color=SURF, lw=4)
for gi, (_, dname) in enumerate(DIRS):
    ax.text(gi * len(METRICS) + (len(METRICS) - 1) / 2, -0.7, dname, ha="center", va="bottom", fontsize=9,
            color=INK)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.outline.set_visible(False)
cb.set_label("relative score (1 = best, 0 = worst non-reference)", fontsize=7.5, color=INK2)
ax.set_title("No model wins everywhere: RAVI variants lead on transfer, LAM on action alignment",
             loc="left", fontsize=10.5, color=INK, pad=30)
fig.text(0.01, -0.16, "Cells show raw means over 5 folds; shading is scaled per column among the four non-reference "
         "models, so small absolute gaps (e.g. forward error 0.219 vs 0.230) can look large.", fontsize=7.5, color=INK2)
save(fig, "fig_scorecard_heatmap")

# ---------------------------------------------------------------- 3. few-shot adaptation curves
ks = [0, 1, 3, 5]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=False)
for ax, (d, dname) in zip(axes, DIRS):
    for m, name, col in MODELS:
        Y = np.stack([vals(d, m, f"fewshot:{k}") for k in ks], 1)
        mu, sd = Y.mean(0), Y.std(0)
        ax.fill_between(ks, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
        ax.plot(ks, mu, color=col, lw=2, marker="o", ms=4.5, mec=SURF, mew=1.2, label=name)
        ax.text(5.15, mu[-1] + {"rave": 0.004, "rave_noanchor": -0.004}.get(m, 0), f"{mu[-1]:.3f}", va="center",
                fontsize=7, color=INK2)
    ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
    ax.text(-0.25, 1.0, "predict no change", va="top" if d == "I" else "bottom", ha="left", fontsize=7, color=INK2)
    ax.set_xticks(ks); ax.set_xlabel("adaptation episodes of the opposite task (k)")
    ax.set_title(dname, loc="left", fontsize=9.5, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(-0.3, 5.9)
axes[0].set_ylabel("forward error on opposite task\n(lower is better)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, -0.1))
fig.suptitle("Koopman variants adapt best when only the IDM is fine-tuned", x=0.01, ha="left", fontsize=11, y=1.03)
fig.tight_layout()
save(fig, "fig_fewshot_curves")

# ---------------------------------------------------------------- 4. per-fold reversibility AUROC (diverging at chance)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0))
norm = TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=0.85)
for ax, (d, dname) in zip(axes, DIRS):
    M = np.full((len(rows), 5), np.nan)
    for i, (m, _, _) in enumerate(rows):
        for r in runs[(d, m)]:
            M[i, r["fold"]] = r["reversibility_auroc"]
    im = ax.imshow(M, cmap=DIVERGE, norm=norm, aspect="auto")
    for i in range(len(rows)):
        for j in range(5):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.6,
                    color="#ffffff" if (v > 0.75 or v < 0.33) else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"fold {k}" for k in range(5)])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for _, n, _ in rows] if d == "I" else [])
    ax.axhline(len(MODELS) - 0.5, color=SURF, lw=4)
    ax.set_title(dname, loc="left", fontsize=9.5, color=INK)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01); cb.outline.set_visible(False)
cb.set_label("AUROC (0.5 = chance)", fontsize=7.5, color=INK2)
fig.suptitle("Reversibility AUROC per fold: sign reversal falls to or below chance", x=0.01, ha="left",
             fontsize=11, y=1.04)
save(fig, "fig_auroc_folds_heatmap")
print("saved to", OUT, sorted(os.listdir(OUT)))
