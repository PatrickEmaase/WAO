# WAO: Learning Reverse Action from Video

Journal draft (JCSE, KIISE) and code for learning reverse actions from forward-only video with a drift-aware
Koopman operator. A latent action `z` acts on a lifted state through

```
K(z) = exp(G0 + sum_i z_i G_i)        G0 = drift, G_i = action generators
backward step:          s_t = K(z)^-1 s_{t+1}
time-reversal action:   z_bar  = -z            (under drift -G0)
undo action:            z_undo = -z - 2c       (c: drift-equivalent action on visited states)
reversibility score:    kappa_t = cond K(z_t)
```

## Layout

| Path | Contents |
| --- | --- |
| `paper/wao.tex`, `paper/wao.bib`, `paper/wao.pdf` | Journal draft (JCSE template, 11 pages) |
| `paper/figs/` | Figures used in the draft |
| `paper/jcse.cls`, `jcse.bst`, `setup.tex`, `geometry.sty`, `logo.eps` | Official JCSE LaTeX package |
| `wao_pipeline.mp4` | Every model variable per step on a held-out episode |
| `code/` | Feature extraction, training/evaluation, figures, video, toy study |
| `results/` | Per-fold results of the 5-fold cross-validation (50 runs) and `summary.txt` |

## Build the paper

```bash
cd paper && pdflatex wao && bibtex wao && pdflatex wao && pdflatex wao
```

## Reproduce

```bash
cd code
python prep.py                       # DINOv2 features from the ALOHA HDF5 episodes
bash run_all.sh                      # 5 folds x 5 models x 2 directions (quota-aware)
python summarize.py                  # results/summary.json
python figs_wao.py                   # figures
python rave.py --train I --fold 0 --model rave --out viz_results --export
python viz_wao.py viz/export_I_0.npz # pipeline video
```

In the code, the model is named `rave` (the earlier working name of WAO).
