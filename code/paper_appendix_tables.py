"""Appendix tables for Revision 1: few-shot at every k (mean +- std over 15 runs) and per-fold values (mean over 3 seeds)."""
import json, glob, sys
from collections import defaultdict
import numpy as np
R = sys.argv[1]
runs = defaultdict(list)
for f in glob.glob(f"{R}/[IR]_*.json"):
    r = json.load(open(f)); runs[(r["train"], r["model"])].append(r)
NM = [("lam", "LAM"), ("signrev", "Sign-rev. LAM"), ("rave_noanchor", "\\method{} w/o anchor"), ("rave", "\\method"),
      ("oracle", "Two-dir. ref.")]


def v(r, k):
    return r["fewshot_fwd_rel_err"][k[3:]] if k.startswith("fs:") else r[k]


print("% ===== FEWSHOT")
ks = ["0", "1", "3", "5"]
cells = {}
for m, _ in NM[:4]:
    for d in "IR":
        for k in ks:
            x = np.array([v(r, "fs:" + k) for r in runs[(d, m)]]); cells[(m, d, k)] = (x.mean(), x.std())
for m, name in NM[:4]:
    row = []
    for d in "IR":
        for k in ks:
            mu, sd = cells[(m, d, k)]
            best = min(cells[(mm, d, k)][0] for mm, _ in NM[:4])
            s = f"{mu:.3f}"; s = f"\\textbf{{{s}}}" if round(mu, 3) == round(best, 3) else s
            row.append(f"{s}$\\pm${sd:.3f}")
    print(f"\t\t{name.replace(chr(92)+'method{}', '{'+chr(92)+'method}').replace(chr(92)+'method', '{'+chr(92)+'method}') if False else name} & " + " & ".join(row) + " \\\\")

for d in "IR":
    print(f"\n% ===== PERFOLD {d}")
    for key, lab, arrow, models in (("end_dist_ratio", "End-distance ratio", "\\downarrow", NM),
                                    ("reversibility_auroc", "Reversibility AUROC", "\\uparrow", NM),
                                    ("action_cos", "Action alignment", "\\uparrow", NM),
                                    ("fs:5", "Few-shot error, $k=5$", "\\downarrow", NM[:4])):
        for i, (m, name) in enumerate(models):
            fm = [np.mean([v(r, key) for r in runs[(d, m)] if r["fold"] == f]) for f in range(5)]
            lead = f"\\multirow{{{len(models)}}}{{*}}{{{lab} ${arrow}$}}" if i == 0 else ""
            print(f"{lead} & {name} & " + " & ".join(f"{x:.3f}" for x in fm) + " \\\\")
        print("\\midrule")
    sr = [np.mean([r["end_dist_ratio"] for r in runs[(d, "signrev")] if r["fold"] == f]) for f in range(5)]
    kmax = max(np.mean([r["end_dist_ratio"] for r in runs[(d, m)] if r["fold"] == f])
               for m in ("rave_noanchor", "rave", "oracle", "rave_sr", "rave_norev", "rave_nodrift") for f in range(5))
    print(f"% signrev fold-mean end min {min(sr):.3f}; Koopman fold-mean max {kmax:.3f}")
    print("% stage reach of n test episodes per fold:", [sum(1 for r in runs[(d, 'rave')] if r['fold'] == f) for f in range(5)])
