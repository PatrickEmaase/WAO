"""Print LaTeX table rows for Revision 1 from results/summary.json and results/paired_stats.json (mean +- std over 5 folds x 3 seeds)."""
import json, sys
R = sys.argv[1]
S = json.load(open(f"{R}/summary.json"))
P = json.load(open(f"{R}/paired_stats.json"))


def g(d, m, k):
    return S.get(f"{d}/{m}/{k}")


def ms(d, m, k, nd=3, bold=False):
    v = g(d, m, k)
    if v is None:
        return "--"
    mu = f"{v[0]:.{nd}f}"
    if bold:
        mu = f"\\textbf{{{mu}}}"
    return f"{mu}$\\pm${v[1]:.{nd}f}"


def mu(d, m, k, nd=3):
    v = g(d, m, k)
    return "--" if v is None else f"{v[0]:.{nd}f}"


print("% n per cell:", {k: v[2] for k, v in S.items() if k.endswith("/end_dist_ratio")})
# ---------------- main table
MAIN = ["lam", "signrev", "rave_noanchor", "rave"]
rows = [("Forward error $\\downarrow$", "forward_rel_err", -1, 3), ("Backward error $\\downarrow$", "backward_rel_err", -1, 3),
        ("End-distance ratio $\\downarrow$", "end_dist_ratio", -1, 3), ("Stage reach $\\uparrow$", "stage_reach", 1, 2),
        ("Action alignment $\\uparrow$", "action_cos", 1, 3), ("Joint-probe error $\\downarrow$", "joint_probe_nmse", -1, 3),
        ("Reversibility AUROC $\\uparrow$", "reversibility_auroc", 1, 3), ("Few-shot error, $k=5$ $\\downarrow$", "fewshot_5", -1, 3),
        ("Drift residual $\\varepsilon$", "drift_residual", 0, 3)]
print("\n% ===== MAIN")
for name, k, sign, nd in rows:
    for i, d in enumerate(("I", "R")):
        vals = {m: g(d, m, k) for m in MAIN}
        best = None
        if sign:
            cand = {m: round(v[0], nd) for m, v in vals.items() if v}
            bv = (max if sign > 0 else min)(cand.values())
            best = {m for m, v in cand.items() if v == bv}
        cells = [ms(d, m, k, nd, best is not None and m in best) for m in MAIN] + [ms(d, "oracle", k, nd)]
        lead = f"\\multirow{{2}}{{*}}{{{name}}}" if i == 0 else ""
        dn = "I$\\to$R" if d == "I" else "R$\\to$I"
        print(f"\t\t{lead} & {dn} & " + " & ".join(cells) + " \\\\")

# ---------------- paired table
print("\n% ===== PAIRED")
NAMES = {"lam": "LAM", "signrev": "Sign-rev. LAM", "rave_noanchor": "{\\method} w/o anchor", "rave_sr": "{\\method}-SR",
         "rave_norev": "{\\method} w/o reversal", "rave_nodrift": "{\\method}, $G_0=0$"}
MN = [("end_dist_ratio", "End-distance ratio $\\downarrow$"), ("reversibility_auroc", "Reversibility AUROC $\\uparrow$"),
      ("fewshot_5", "Few-shot error, $k=5$ $\\downarrow$"), ("action_cos", "Action alignment $\\uparrow$")]
idx = {(p["dir"], p["other"], p["metric"]): p for p in P}


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


for o in ["lam", "signrev", "rave_noanchor", "rave_sr", "rave_norev", "rave_nodrift"]:
    for j, (k, name) in enumerate(MN):
        c = []
        for d in ("I", "R"):
            p = idx[(d, o, k)]
            c += [f"{p['diff_mean']:+.3f}$\\pm${p['diff_std']:.3f}", f"{p['better']}/{p['n']}", pv(p["p_t"])]
        lead = f"\\multirow{{4}}{{*}}{{{NAMES[o]}}}" if j == 0 else ""
        print(f"\t\t{lead} & {name} & " + " & ".join(c) + " \\\\")
    print("\t\t\\midrule")

# ---------------- ablation table
print("\n% ===== ABLATION")
AB = [("rave", "{\\method}"), ("rave_noanchor", "{\\method} w/o anchor"), ("rave_sr", "{\\method}-SR ($+G_0$)"),
      ("rave_norev", "{\\method} w/o reversal"), ("rave_nodrift", "{\\method}, $G_0=0$")]
for m, name in AB:
    c = []
    for k in ("end_dist_ratio", "reversibility_auroc", "fewshot_5", "absorption_probe_auroc"):
        c += [ms("I", m, k), ms("R", m, k)]
    print(f"\t\t{name} & " + " & ".join(c) + " \\\\")
print("\t\t{\\method}, $G_0$ fine-tuned & -- & -- & -- & -- & " + ms("I", "rave", "fewshot_g0_5") + " & " + ms("R", "rave", "fewshot_g0_5") + " & -- & -- \\\\")
for m, name in (("lam", "LAM"), ("signrev", "Sign-rev. LAM")):
    c = [ms("I", m, "end_dist_ratio"), ms("R", m, "end_dist_ratio"),
         ms("I", m, "reversibility_auroc") + "$^\\dagger$", ms("R", m, "reversibility_auroc") + "$^\\dagger$",
         ms("I", m, "fewshot_5"), ms("R", m, "fewshot_5"), ms("I", m, "absorption_probe_auroc"), ms("R", m, "absorption_probe_auroc")]
    print(f"\t\t{name} & " + " & ".join(c) + " \\\\")

# ---------------- detection table (means only)
print("\n% ===== DETECT")
DT = [("$\\kappa_t$ ({\\method})", "rave", "kappa", True, True), ("$\\klin_t$ ({\\method})", "rave", "kappa_lin", True, True),
      ("$\\rho_t$ ({\\method})", "rave", "rho", True, True), ("$\\varepsilon_t$ ({\\method})", "rave", "eps_t", True, True),
      ("Jacobian cond.\\ (LAM)$^\\dagger$", "lam", "kappa", True, True), None,
      ("$\\|z_t\\|$ ({\\method})", "rave", "znorm", True, True), ("$\\|o_{t+1}-o_t\\|$", "rave", "feat_change", True, False),
      ("Arm joint speed", "rave", "arm_speed", False, True), ("Gripper opening", "rave", "gripper_open", True, True)]
for r in DT:
    if r is None:
        print("\t\t\\midrule"); continue
    name, m, k, mm, dm = r
    c = [mu("I", m, f"auroc_{k}"), mu("R", m, f"auroc_{k}"),
         mu("I", m, f"auroc_mm_{k}") if mm else "--", mu("R", m, f"auroc_mm_{k}") if mm else "--",
         mu("I", m, f"auroc_dm_{k}") if dm else "--", mu("R", m, f"auroc_dm_{k}") if dm else "--"]
    print(f"\t\t{name} & " + " & ".join(c) + " \\\\")

# ---------------- undo table
print("\n% ===== UNDO")
def two(k, nd=3):
    return mu("I", "rave", k, nd) + " & " + mu("R", "rave", k, nd)
def pair(k1, k2, nd=3, fmt="{a} / {b}"):
    return " & ".join(fmt.format(a=mu(d, "rave", k1, nd), b=mu(d, "rave", k2, nd)) for d in ("I", "R"))
print("\t\tEnd-distance ratio, {\\method} & " + two("end_dist_ratio") + " \\\\")
print("\t\t\\quad null: zero actions & " + two("end_dist_ratio_null_zero") + " \\\\")
print("\t\t\\quad null: random actions & " + two("end_dist_ratio_null_random") + " \\\\")
print("\t\t\\quad null: unreversed forward actions & " + two("end_dist_ratio_null_forward") + " \\\\")
print("\t\tAction alignment, {\\method} & " + two("action_cos") + " \\\\")
print("\t\t\\quad reference: time-permuted actions & " + two("action_cos_permuted") + " \\\\")
print("\t\t\\midrule")
print("\t\tDrift residual $\\varepsilon$, {\\method} & " + two("drift_residual") + " \\\\")
print("\t\t\\quad analytic reference (Remark~\\ref{r:eps}) & $\\ge0.87$ & $\\ge0.87$ \\\\")
print("\t\t\\quad random $G_0$, matched norm & " + two("drift_residual_randomG0") + " \\\\")
print("\t\t$\\varepsilon_t$: post-release / carrying & " + pair("eps_t_release_mean", "eps_t_carry_mean") + " \\\\")
print("\t\tUndo success, one step & " + pair("undo1_success_release", "undo1_success_carry", 2) + " \\\\")
print("\t\tUndo success, three steps & " + pair("undoH_success_release", "undoH_success_carry", 2) + " \\\\")
print("\t\t$\\|A(z_t)\\|_2$: median (max) & " + pair("A_norm_median", "A_norm_max", 2, "{a} ({b})") + " \\\\")

# ---------------- other models' undo for the text
print("\n% ===== UNDO OTHER MODELS (I rel/carry, R rel/carry)")
for m in ["lam", "signrev", "rave_noanchor", "rave", "oracle", "rave_sr", "rave_norev", "rave_nodrift"]:
    print(m, [mu(d, m, k, 3) for d in ("I", "R") for k in ("undo1_success_release", "undo1_success_carry",
                                                         "undoH_success_release", "undoH_success_carry")])
