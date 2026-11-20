"""Aggregate results/*.json into mean +- std over folds (and seeds), per direction and model.

Revision 1: reads every seed (files *_s<seed>.json), adds the three Koopman ablations and prints the new diagnostics
in a second block. n in summary.json counts fold x seed runs.
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results") if glob.glob(os.path.join(ROOT, "results", "[IR]_*.json")) \
    else os.path.join(ROOT, "..", "results")
PRE = sys.argv[1] if len(sys.argv) > 1 else ""      # revision 2: "k10_" summarises the 10-fold runs
MODELS = ["lam", "signrev", "rave_noanchor", "rave", "oracle", "rave_sr", "rave_norev", "rave_nodrift"]
METRICS = ["forward_rel_err", "backward_rel_err", "stage_reach", "end_dist_ratio", "action_cos",
           "joint_probe_nmse", "joint_negation_nmse", "reversibility_auroc", "drift_residual"]
EXTRA = ["end_dist_ratio_null_zero", "end_dist_ratio_null_random", "end_dist_ratio_null_forward",
         "end_dist_ratio_negz", "end_dist_ratio_undo", "action_cos_permuted",
         "auroc_kappa", "auroc_kappa_lin", "auroc_rho", "auroc_eps_t", "auroc_znorm", "auroc_feat_change",
         "auroc_arm_speed", "auroc_gripper_open",
         "auroc_mm_kappa", "auroc_mm_kappa_lin", "auroc_mm_rho", "auroc_mm_eps_t", "auroc_mm_znorm",
         "auroc_mm_feat_change", "auroc_mm_arm_speed", "auroc_mm_gripper_open",
         "auroc_dm_kappa", "auroc_dm_kappa_lin", "auroc_dm_rho", "auroc_dm_eps_t", "auroc_dm_znorm",
         "auroc_dm_arm_speed", "auroc_dm_gripper_open",
         "drift_residual_randomG0", "eps_t_release_mean", "eps_t_carry_mean",
         "undo_closed_relerr_release", "undo_closed_relerr_carry", "undo1_success_release", "undo1_success_carry",
         "undoH_success_release", "undoH_success_carry", "A_norm_median", "A_norm_max", "strip_frac",
         "absorption_probe_auroc", "absorption_probe_auroc_dfeat",
         # revision 2
         "end_dist_rel_null", "end_dist_ratio_decstart", "end_dist_ratio_null_zero_decstart",
         "absorption_probe_auroc_resid", "absorption_probe_auroc_motion",
         "auroc_inv_err", "auroc_mm_inv_err", "auroc_dm_inv_err", "auroc_inv_err_abs", "auroc_mm_inv_err_abs",
         "auroc_dm_inv_err_abs", "undo_closed_abserr_release", "undo_closed_abserr_carry", "undo1_abserr_release",
         "undo1_abserr_carry", "undoH_abserr_release", "undoH_abserr_carry", "step_len_release", "step_len_carry"]


def flat(r):
    out = dict(r)
    for k in (0, 1, 3, 5):
        for src, dst in (("fewshot_fwd_rel_err", "fewshot"), ("fewshot_fwd_rel_err_tuneG0", "fewshot_g0")):
            if src in r:
                out[f"{dst}_{k}"] = r[src][str(k)]
    return out


runs = defaultdict(list)
for f in glob.glob(os.path.join(RES, PRE + "[IR]_*.json")):                          # skip summary.json
    r = flat(json.load(open(f))); runs[(r["train"], r["model"])].append(r)

summary = {}
for direction in ("I", "R"):
    present = [m for m in MODELS if runs.get((direction, m))]
    for block, mets in (("main", METRICS + [f"fewshot_{k}" for k in (0, 1, 3, 5)] + [f"fewshot_g0_{k}" for k in (0, 1, 3, 5)]),
                        ("revision-1 diagnostics", EXTRA)):
        print(f"\n=== trained on {'insertion' if direction == 'I' else 'removal'} ({direction}), {block} ===")
        print(f"{'metric':30s}" + "".join(f"{m:>18s}" for m in present))
        for met in mets:
            row, any_ = f"{met:30s}", False
            for mdl in present:
                vals = [r[met] for r in runs[(direction, mdl)] if r.get(met) is not None]
                if vals:
                    any_ = True
                    summary[f"{direction}/{mdl}/{met}"] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
                    row += f"{np.mean(vals):11.3f}±{np.std(vals):.3f}"
                else:
                    row += f"{'-':>18s}"
            if any_:
                print(row)
json.dump(summary, open(os.path.join(RES, f"summary{PRE.rstrip(chr(95)) and chr(95) + PRE.rstrip(chr(95))}.json"), "w"), indent=1)
