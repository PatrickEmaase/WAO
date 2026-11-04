"""Build RAVI_Main_2026.tex: ravi.tex without appendices, with every in-text pointer to them removed."""
import re

t = open("ravi.tex", encoding="utf-8").read()

# 1. drop everything from the page break before the appendix up to \end{document}
a = t.index("\\clearpage\n% =====================================================================\n\\appendix")
b = t.index("\\end{document}")
t = t[:a] + t[b:]

# 2. rewrite in-text pointers to the appendices
rep = [
    (", supported in the appendices by per-fold results, a controlled two-dimensional study, development runs and a step-by-step walkthrough of every model variable.",
     "."),
    ("Section~\\ref{s:discussion} discusses limitations and Section~\\ref{s:conclusion} concludes; the appendices give further results, a controlled study, development runs and implementation details.",
     "Section~\\ref{s:discussion} discusses limitations, and Section~\\ref{s:conclusion} concludes."),
    ("The design is task-independent; the instantiation used in our experiments is shown in \\ref{a:impl} (Fig.~\\ref{f:pipeline}).",
     "The design is task-independent; Section~\\ref{s:setup} describes the instantiation used in our experiments."),
    ("(chance 33.3\\%; \\ref{a:dev}).", "(chance 33.3\\%)."),
]
for old, new in rep:
    assert old in t, old[:60]
    t = t.replace(old, new)

s = t.index("\\subsection{Further Results}")
e = t.index("% =====================================================================\n\\section{Discussion and Limitations}")
t = t[:s] + ("\\subsection{Joint Motion}\n"
             "No model recovers the real reverse joint motion: every joint-probe error exceeds 1.0, although all beat "
             "negated, time-reversed joint deltas (1.58 and 1.53), which confirms that removal is not insertion played backward.\n\n\n") + t[e:]

t = t.replace("% Build: pdflatex ravi; bibtex ravi; pdflatex ravi; pdflatex ravi",
              "% Main paper without appendices.\n% Build: pdflatex RAVI_Main_2026; bibtex RAVI_Main_2026; pdflatex RAVI_Main_2026; pdflatex RAVI_Main_2026")

# 3. nothing may still point at appendix-only labels
gone = ["a:cv", "a:toy", "a:dev", "a:impl", "f:pipeline", "f:scorecard", "f:walk", "t:fewshot", "t:perfoldI",
        "t:perfoldR", "t:toy", "t:dev", "t:feat"]
left = [g for g in gone if "{" + g + "}" in t]
assert not left, left
body = "\n".join(l for l in t.splitlines() if not l.lstrip().startswith("%"))
assert "\\appendix" not in body and "ppendi" not in body, re.findall(r".{0,40}ppendi.{0,40}", body)
open("RAVI_Main_2026.tex", "w", encoding="utf-8").write(t)
print("wrote RAVI_Main_2026.tex", len(t.splitlines()), "lines")
