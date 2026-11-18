"""Paired comparisons of RAVI with every other model over (fold, seed) pairs; writes results/paired_stats.{json,tex}.

All models of one direction share the test folds, so differences are paired. With 5 folds and one seed a two-sided
Wilcoxon signed-rank test cannot go below p = 0.0625, so we report the paired t-test, the Wilcoxon test (exact) and the
number of pairs in which RAVI is better.
"""
import glob, json, os
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results") if glob.glob(os.path.join(ROOT, "results", "[IR]_*.json")) \
    else os.path.join(ROOT, "..", "results")
REF = "rave"
OTHERS = [("lam", "LAM"), ("signrev", "Sign-rev. LAM"), ("rave_noanchor", "{\\method} w/o anchor"),
          ("rave_sr", "{\\method}-SR"), ("rave_norev", "{\\method} w/o reversal"), ("rave_nodrift", "{\\method}, $G_0=0$")]
METRICS = [("end_dist_ratio", "End-distance ratio", -1), ("reversibility_auroc", "Reversibility AUROC", 1),
           ("fewshot_5", "Few-shot error, $k=5$", -1), ("action_cos", "Action alignment", 1),
           ("forward_rel_err", "Forward error", -1)]

R = defaultdict(dict)
for f in glob.glob(os.path.join(RES, "[IR]_*.json")):
    r = json.load(open(f)); r["fewshot_5"] = r.get("fewshot_fwd_rel_err", {}).get("5")
    R[(r["train"], r["model"])][(r["fold"], r.get("seed", 0))] = r

out, tex = [], []
for d in ("I", "R"):
    for o, oname in OTHERS:
        if (d, o) not in R:
            continue
        for met, mname, sign in METRICS:
            keys = sorted(k for k in R[(d, REF)] if k in R[(d, o)]
                          and R[(d, REF)][k].get(met) is not None and R[(d, o)][k].get(met) is not None)
            if len(keys) < 3:
                continue
            a = np.array([R[(d, REF)][k][met] for k in keys]); b = np.array([R[(d, o)][k][met] for k in keys])
            diff = a - b; better = int(((diff * sign) > 0).sum())
            t = stats.ttest_rel(a, b).pvalue
            w = stats.wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
            out.append(dict(dir=d, other=o, metric=met, n=len(keys), diff_mean=float(diff.mean()),
                            diff_std=float(diff.std(ddof=1)), better=better, p_t=float(t), p_wilcoxon=float(w)))
            dname = "I$\\to$R" if d == "I" else "R$\\to$I"   # outside the f-string (Python < 3.12)
            tex.append(f"{dname} & {oname} & {mname} & "
                       f"{diff.mean():+.3f}$\\pm${diff.std(ddof=1):.3f} & {better}/{len(keys)} & {t:.3f} \\\\")
            print(f"{d} {o:14s} {met:20s} RAVI-other {diff.mean():+.3f}±{diff.std(ddof=1):.3f}  "
                  f"RAVI better {better}/{len(keys)}  p_t={t:.3f}  p_W={w:.3f}")
json.dump(out, open(os.path.join(RES, "paired_stats.json"), "w"), indent=1)
open(os.path.join(RES, "paired_stats.tex"), "w").write("\n".join(tex) + "\n")
