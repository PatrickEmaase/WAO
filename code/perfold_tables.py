"""Write LaTeX per-fold tables (appendix) from results/[IR]_<fold>_<model>.json."""
import glob, json, os, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else "perfold_tables.tex"
MODELS = [("lam", "LAM"), ("signrev", "Sign-rev. LAM"), ("rave_noanchor", "\\method{} w/o anchor"),
          ("rave", "\\method"), ("oracle", "Oracle")]
METRICS = [("end_dist_ratio", "End-distance ratio $\\downarrow$", "{:.3f}"),
           ("reversibility_auroc", "Reversibility AUROC $\\uparrow$", "{:.3f}"),
           ("action_cos", "Action alignment $\\uparrow$", "{:.3f}"),
           ("fewshot:5", "Few-shot error, $k=5$ $\\downarrow$", "{:.3f}")]

R = {}
for f in glob.glob(os.path.join(ROOT, "[IR]_*.json")):
    r = json.load(open(f)); R[(r["train"], r["fold"], r["model"])] = r


def val(r, key):
    if key.startswith("fewshot:"):
        v = r.get("fewshot_fwd_rel_err", {}).get(key.split(":")[1])
    else:
        v = r.get(key)
    return v


lines = []
for d, dname in (("I", "trained on insertion, tested on removal (I$\\to$R)"),
                 ("R", "trained on removal, tested on insertion (R$\\to$I)")):
    lines += ["\\begin{table*}", "\\centering",
              f"\\caption{{Per-fold results, {dname}. Folds are disjoint sets of test episodes.}}",
              f"\\label{{t:perfold{d}}}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
              "\\begin{tabular}{llccccc}", "\\toprule",
              "Metric & Model & Fold 0 & Fold 1 & Fold 2 & Fold 3 & Fold 4 \\\\", "\\midrule"]
    for key, mname, fmt in METRICS:
        rows = [(m, n) for m, n in MODELS if not (key.startswith("fewshot") and m == "oracle")]
        for j, (m, n) in enumerate(rows):
            cells = []
            for fold in range(5):
                r = R.get((d, fold, m)); v = val(r, key) if r else None
                cells.append(fmt.format(v) if v is not None else "--")
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{mname}}}" if j == 0 else ""
            lines.append(f"{first} & {n} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if key != METRICS[-1][0] else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table*}", ""]
open(OUT, "w").write("\n".join(lines))
print("wrote", OUT, "runs", len(R))
