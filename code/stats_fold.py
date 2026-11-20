"""Revision 2: fold-level paired comparisons of RAVI with every other model, Holm-corrected across the whole table.

Seeds are averaged within each fold, so the unit of analysis is the fold (the independent test data). For a balanced
design this paired t-test on fold means is identical to a random-intercept mixed model with fold as random effect.
Usage: python stats_fold.py [prefix]   prefix "" = 5-fold x 3 seeds (results/[IR]_*.json), "k10_" = 10-fold runs.
Writes results/paired_fold<prefix>.{json,tex,txt}.
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "..", "results")
PRE = sys.argv[1] if len(sys.argv) > 1 else ""
REF = "rave"
OTHERS = [("lam", "LAM"), ("signrev", "Sign-rev. LAM"), ("rave_noanchor", "{\\method} w/o anchor"),
          ("rave_sr", "{\\method}-SR ($+G_0$)"), ("rave_norev", "{\\method} w/o reversal"), ("rave_nodrift", "{\\method}, $G_0=0$")]
METRICS = [("end_dist_ratio", "End-distance ratio $\\downarrow$", -1), ("reversibility_auroc", "Reversibility AUROC $\\uparrow$", 1),
           ("fewshot_5", "Few-shot error, $k=5$ $\\downarrow$", -1)]
EXTRA_METRICS = [("end_dist_rel_null", "End distance / own null $\\downarrow$", -1), ("action_cos", "Action alignment", 1)]

pat = re.compile(r"^" + re.escape(PRE) + r"([IR])_(\d+)_([a-z_]+?)(?:_s(\d+))?\.json$")
R = defaultdict(lambda: defaultdict(list))                       # (dir, model) -> fold -> [run dicts]
for f in glob.glob(os.path.join(RES, PRE + "[IR]_*.json")):
    m = pat.match(os.path.basename(f))
    if not m:
        continue
    r = json.load(open(f)); r["fewshot_5"] = r.get("fewshot_fwd_rel_err", {}).get("5")
    R[(m.group(1), m.group(3))][int(m.group(2))].append(r)


def fold_means(d, model, met):
    out = {}
    for fold, runs in R[(d, model)].items():
        v = [x[met] for x in runs if x.get(met) is not None]
        if v:
            out[fold] = float(np.mean(v))
    return out


rows = []
for d in ("I", "R"):
    for o, oname in OTHERS:
        for met, mname, sign in METRICS + EXTRA_METRICS:
            a, b = fold_means(d, REF, met), fold_means(d, o, met)
            folds = sorted(set(a) & set(b))
            if len(folds) < 3:
                continue
            x, y = np.array([a[f] for f in folds]), np.array([b[f] for f in folds]); diff = x - y
            p_t = float(stats.ttest_rel(x, y).pvalue)
            p_w = float(stats.wilcoxon(x, y).pvalue) if np.any(diff != 0) else 1.0
            rows.append(dict(dir=d, other=o, oname=oname, metric=met, mname=mname, n_folds=len(folds),
                             seeds_per_fold=float(np.mean([len(R[(d, REF)][f]) for f in folds])),
                             diff_mean=float(diff.mean()), diff_std=float(diff.std(ddof=1)),
                             wins=int(((diff * sign) > 0).sum()), p_t=p_t, p_wilcoxon=p_w,
                             in_table=met in [m for m, _, _ in METRICS]))

# Holm correction across the whole Table II family (3 metrics x 6 comparisons x 2 directions)
fam = [r for r in rows if r["in_table"]]
order = np.argsort([r["p_t"] for r in fam]); m = len(fam); running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * fam[i]["p_t"]))
    fam[i]["p_holm"] = running
for r in rows:
    r.setdefault("p_holm", None)

json.dump(rows, open(os.path.join(RES, f"paired_fold{PRE.rstrip('_') and '_' + PRE.rstrip('_')}.json"), "w"), indent=1)
pf = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
lines, txt = [], []
for o, oname in OTHERS:
    for j, (met, mname, _) in enumerate(METRICS):
        c = []
        for d in ("I", "R"):
            r = next((r for r in fam if r["dir"] == d and r["other"] == o and r["metric"] == met), None)
            c += ["--"] * 3 if r is None else [f"{r['diff_mean']:+.3f}$\\pm${r['diff_std']:.3f}", f"{r['wins']}/{r['n_folds']}", pf(r["p_holm"])]
        lead = f"\\multirow{{3}}{{*}}{{{oname}}}" if j == 0 else ""
        lines.append(f"\t\t{lead} & {mname} & " + " & ".join(c) + " \\\\")
    lines.append("\t\t\\midrule")
open(os.path.join(RES, f"paired_fold{PRE.rstrip('_') and '_' + PRE.rstrip('_')}.tex"), "w").write("\n".join(lines[:-1]) + "\n")
for r in rows:
    txt.append(f"{r['dir']} {r['other']:14s} {r['metric']:22s} n={r['n_folds']:2d} d={r['diff_mean']:+.3f}+-{r['diff_std']:.3f} "
               f"wins={r['wins']:2d} p_t={r['p_t']:.4f} p_W={r['p_wilcoxon']:.4f} p_holm={'' if r['p_holm'] is None else f'{r[chr(112)+chr(95)+chr(104)+chr(111)+chr(108)+chr(109)]:.4f}'}")
open(os.path.join(RES, f"paired_fold{PRE.rstrip('_') and '_' + PRE.rstrip('_')}.txt"), "w").write("\n".join(txt) + "\n")
print("\n".join(txt))
