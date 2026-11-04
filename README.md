# WAO: Learning Reverse Actions from Video

WAO learns latent action dynamics from forward-only robot videos and derives reverse actions in closed form.
A latent action `z` acts on a lifted state through a Koopman operator with a separate drift generator:

```
K(z) = exp(G0 + sum_i z_i G_i),   s_{t+1} = K(z) s_t
backward step:          s_t = K(z)^-1 s_{t+1} = exp(-G0 - sum_i z_i G_i) s_{t+1}
time-reversal action:   z_bar  = -z   (with drift -G0)
undo action:            z_undo = -z - 2c,   sum_i c_i G_i ~ G0
reversibility score:    kappa_t = cond K(z_t)
```

Paper: *Learning Reverse Actions from Video*, P. M. Emaase, S. Choi and B.-T. Zhang (JCSE draft, `paper/wao.pdf`).

## Data

ALOHA bimanual toaster task (not included in this repo):

- `toaster_forward/episode_*.hdf5`: 51 insertion episodes (bread from plate into toaster); one static recording is excluded, leaving 50
- `toaster_backward/episode_*.hdf5`: 50 removal episodes (bread from toaster back to plate)

Each episode is 1,250 steps at 50 Hz with `cam_high`, `cam_left_wrist` and `cam_right_wrist` (480x848, JPEG).

## Files

| File | Purpose |
| --- | --- |
| `code/prep.py` | Decode `cam_high` at 10 Hz and extract DINOv2 ViT-S/14 features into `toaster_{forward,backward}.npz` |
| `code/prep_dir.py` | Same as `prep.py` for directories given on the command line |
| `code/rave.py` | Train and evaluate one model / direction / fold (`lam`, `signrev`, `rave_noanchor`, `rave`, `oracle`); `rave` is WAO |
| `code/run_all.sh`, `code/run_one.sh` | 5-fold cross-validation in both directions for all models, within the GPU quota |
| `code/summarize.py` | Aggregate `results/*.json` into the paper tables |
| `code/perfold_tables.py` | Per-fold LaTeX tables for the appendix |
| `code/figs_wao.py` | Figures (key metrics, scorecard, few-shot curves, per-fold AUROC) |
| `code/viz_wao.py` | Video of every model variable on a held-out episode |
| `code/feat_check.py` | Stage-decoding feature check |
| `code/toy_bik_lam.py` | 2-D toy check of the drift-aware reverse action |
| `results/` | Per-fold results of the 5-fold cross-validation (50 runs), `summary.txt`, feature check |
| `paper/` | Journal draft (JCSE template), figures, bibliography |
| `wao_pipeline.mp4` | Pipeline video |

In the code, WAO is still named `rave` (its earlier working name); the model keys in `results/` follow the code.

## Run

```bash
cd code
python prep.py
bash run_all.sh
python summarize.py
python figs_wao.py
```

Pipeline video for one held-out episode:

```bash
python rave.py --train I --fold 0 --model rave --out viz_results --export
python viz_wao.py viz/export_I_0.npz
```

Build the paper:

```bash
cd paper && pdflatex wao && bibtex wao && pdflatex wao && pdflatex wao
```
