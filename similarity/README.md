# Similarity check for the RAVI journal papers

This folder holds an in-house, iThenticate-style overlap check. It is **not** a substitute for the official
check (iThenticate / Crossref Similarity Check or the KCI similarity check); run that before submission.

## Method

- `simcheck.py paper.txt sources/ 8`: share of the paper's words that lie inside 8-word sequences also found in a
  source (lower-cased, punctuation removed). Reported for the whole PDF and for the body with the reference list excluded.
- `flaglines.py paper.tex sources/`: the same test per LaTeX source line, used to find passages to reword.
- Comparison corpus (`sources/`, not committed):
  - our own earlier RAVE conference drafts (`main.tex`, `RAVE_KIISE_V1`, `RAVE_KIISE_V2`) and the RAVE draft document;
  - the arXiv abstracts of 23 of the cited works (the CVPR arrow-of-time paper and Higham 2005 are not on arXiv);
  - the JCSE open-access licence text that every JCSE paper carries.

## Result

| Paper | Before, whole PDF | After, whole PDF | After, body (references excluded) |
| --- | --- | --- | --- |
| `ravi.pdf` (14 pages, with appendices) | 24.6% | 7.1% | **3.2%** |
| `RAVI_Main_2026.pdf` (10 pages) | 30.0% | 9.5% | **3.0%** |

Almost all of the original overlap was text reused from our own conference drafts. The cited papers' abstracts
overlap by less than 0.2% each, before and after.

What remains:

- **References** (about 3.9 points of the whole-PDF score for `ravi.pdf`): the bibliography entries are the same
  strings as in the conference drafts and in any other paper citing these works. Similarity reports are normally
  run with the bibliography excluded; with it excluded both papers are near 3%.
- **JCSE licence boilerplate** (0.4 to 0.6 points), shared by every JCSE paper.
- **Result numbers** that necessarily repeat the conference version's values.

## What was changed

- `rewrite1.py`, `rewrite2.py`: 101 rewordings of sentences, captions and headings that matched the conference
  drafts. Numbers, symbols, citations and claims are unchanged. The training objective is now written as named
  terms ($\ell_{\mathrm{fwd}}$, $\ell_{\mathrm{bwd}}$, $\ell_{\mathrm{rev}}$, $\ell_{\mathrm{neg}}$, $\ell_{\mathrm{idle}}$, $\Omega$).
- `code/figs_journal.py`: journal-style figures without in-figure titles, footnotes or value labels (captions carry
  the text; all values are in the tables).
- `apply_lenient.py`: applies the same rewordings to `RAVI_Main_2026.tex`, which has no appendix.

## Disclosure

The journal paper extends our KIISE conference work on the same method. Journals expect this to be disclosed
(for example in the cover letter and a footnote on the first page) even when the text has been rewritten.
