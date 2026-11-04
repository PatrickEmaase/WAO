"""Aggregate results/*.json into mean +- std over folds, per direction and model."""
import glob, json, os
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = ["lam", "signrev", "rave_noanchor", "rave", "oracle"]
METRICS = ["forward_rel_err", "backward_rel_err", "stage_reach", "end_dist_ratio", "action_cos",
           "joint_probe_nmse", "joint_negation_nmse", "reversibility_auroc", "drift_residual"]

runs = defaultdict(list)
for f in glob.glob(os.path.join(ROOT, "results", "[IR]_*.json")):          # skip summary.json
    r = json.load(open(f)); runs[(r["train"], r["model"])].append(r)

summary = {}
for direction in ("I", "R"):
    print(f"\n=== trained on {'insertion' if direction == 'I' else 'removal'} ({direction}) ===")
    print(f"{'metric':24s}" + "".join(f"{m:>18s}" for m in MODELS))
    for met in METRICS + [f"fewshot_{k}" for k in (0, 1, 3, 5)]:
        row = f"{met:24s}"
        for mdl in MODELS:
            rs = runs.get((direction, mdl), [])
            if met.startswith("fewshot_"):
                vals = [r["fewshot_fwd_rel_err"][met.split("_")[1]] for r in rs if "fewshot_fwd_rel_err" in r]
            else:
                vals = [r[met] for r in rs if r.get(met) is not None]
            if vals:
                summary[f"{direction}/{mdl}/{met}"] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
                row += f"{np.mean(vals):11.3f}±{np.std(vals):.3f}"
            else:
                row += f"{'-':>18s}"
        print(row)
json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)
