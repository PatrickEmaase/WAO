# RAVI revision 1: what changed and what still needs a GPU run

Everything here is a real result or an exact protocol fact read from the code. Robot-data cells that need new runs
are red **tbd** / **[TODO]** in `paper/ravi.tex`; nothing was invented.

## Done in this revision
- **Authors**: Patrick Mokodir Emaase (emaase@bi.snu.ac.kr), Choi Suhyung (shchoi@bi.snu.ac.kr), Byoung Tak Zhang
  (btzhang@snu.ac.kr), in `ravi.tex`, `RAVI_Main_2026.tex`, `cover_letter.tex`, README. Short authors: "P.M. Emaase, S. Choi
  and B.T. Zhang". Affiliations unchanged.
- **Theory**: first-order reading of time reversal (Sec. III); state-independent vs state-specific undo (Def. 2,
  Prop. 4(b), new 4(d)); Remark "Reduction of the drift-aware reversal loss"; Remark "Reference value for the drift
  residual"; gauge consequences after Prop. 5; scope note after Prop. 6 (MLP baselines not covered); Corollary
  "What kappa_t measures" (depends on z only; translations raise it). Fig. 1 labels corrected.
- **Protocol facts from the code**, now stated: end distance in decoded 128-D features; AUROC on held-out recordings of
  the *training* direction, all releases (93/810 and 103/898 release/carry steps); the reference rolls out the test
  recording's own IDM actions (renamed "two-direction reference", not an upper bound); z noise, 1e-3 z penalty, clipping.
- **New results from the existing 50 runs** (`code/stats.py`, Table "paired"): RAVI vs LAM not significant on end
  distance or AUROC; RAVI transfers better in 5/5 folds; LAM aligns actions better in 5/5; the anchor raises AUROC in
  every fold (p = 0.004, 0.002) but costs transfer. Figures regenerated with the new label.
- **Controlled study with known dissipative drift** (`code/toy_reversibility.py`, 7 models x 3 seeds, CPU; results in
  `results/toy_reversibility.json`, Sec. VII-E): decoding under -G0 absorbs less drift than +G0 (0.70 vs 0.81); kappa
  separates irreversible steps only through motion (AUROC about 0.5 at equal displacement); the drift correction improves
  the undo action (0.76 vs 1.22 step lengths) but the plain LAM undoes best (0.53); the anchor costs the Koopman fit.
- **Code**: `rave.py` gains `rave_sr`, `rave_norev`, `rave_nodrift`, `--seed` (seed-tagged files), `--device`,
  `--data_dir`; new `extras.py` adds every diagnostic below. Original keys of the 50 runs are reproduced unchanged.
  Smoke-tested end to end on CPU with synthetic data only.

## To run on the GPU server (fills every tbd)
```bash
cd code
SEEDS="0 1 2" bash run_all.sh          # skips the 50 existing runs; 8 models x 2 dirs x 5 folds x 3 seeds
python summarize.py                    # main block + revision-1 diagnostics, writes results/summary.json
python stats.py                        # paired tests -> results/paired_stats.{json,tex}
python figs_journal.py ../results ../paper/figs
```
Note: the original 50 runs predate `extras.py`, so seed 0 of lam/signrev/rave_noanchor/rave/oracle lacks the new keys.
To get them, rerun those 50 with `--seed 0` after moving the old files aside (the original keys come out identical).

| Paper element | Keys in results/*.json |
|---|---|
| Table "ablation" | `end_dist_ratio`, `reversibility_auroc`, `fewshot_fwd_rel_err`, `fewshot_fwd_rel_err_tuneG0`, `absorption_probe_auroc` for `rave_sr`, `rave_norev`, `rave_nodrift` |
| Table "detect" | `auroc_{kappa,kappa_lin,rho,eps_t,znorm,feat_change,arm_speed,gripper_open}`, same with `auroc_mm_` (arm speed) and `auroc_dm_` (feature change) |
| Table "undo" | `end_dist_ratio_null_{zero,random,forward}`, `action_cos_permuted`, `drift_residual_randomG0`, `eps_t_{release,carry}_mean`, `undo1_success_*`, `undoH_success_*`, `A_norm_median`, `A_norm_max` |
| Table I and text | means over folds x seeds; replace the "one training seed" TODOs |

## Author inputs and final pass
- Rewrite the red TODO sentences in Sec. VII-A to VII-D, VIII and the abstract once the tables are filled.
- The local preview PDFs use `ieeetr.bst` (IEEEtran.bst is not installed here); the source still says IEEEtran.
- `cover_letter.pdf` was built without `lmodern` (not installed here); rebuild locally.
- References goswami2017global, proctor2016dmdc and higham2008functions were added to `ravi.bib`.

## Status (19 Nov 2026): all GPU runs done, every tbd filled
- `SEEDS="0 1 2" bash run_all.sh` on dale (GPUs 2 and 5, 3 jobs at a time): 240 runs (8 models x 2 dirs x 5 folds x 3 seeds),
  no failures. Seed 0 of the five original models was rerun with the new keys; all 590 original values are identical.
- `summarize.py` -> `results/summary.{json,txt}`; `stats.py` -> `results/paired_stats.{json,tex,txt}` (15 pairs per test;
  the f-string in `stats.py` was rewritten for Python 3.10); `figs_journal.py` now averages over all seeds.
- `code/paper_tables.py` and `code/paper_appendix_tables.py` print every table row of the paper from the results.
- Every table and number in `ravi.tex` / `RAVI_Main_2026.tex` is now a mean over 15 runs; all red tbd / TODO are gone
  and the text of Sec. VII-A to VII-D, VIII, IX, the abstract and the findings was rewritten from the results.
- Page counts: `RAVI_Main_2026.pdf` 17 pages, `ravi.pdf` 21 pages (JCSE limit used so far: 14).
