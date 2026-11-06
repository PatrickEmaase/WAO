"""iThenticate-style similarity check: share of the paper's words covered by word n-grams that also occur in a source.

Usage: python simcheck.py paper.txt sources_dir [n] [--report out.txt]
- paper.txt: plain text of the paper (pdftotext output)
- sources_dir: one .txt file per source
Prints coverage per source, overall coverage (union), with and without the reference list, and the longest matches.
"""
import os, re, sys
from collections import defaultdict

paper_path, src_dir = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 8
report = sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else None


def tokens(text):
    text = text.replace("’", "'").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"-\n", "", text)                      # rejoin hyphenated line breaks
    return re.findall(r"[a-z0-9]+", text.lower())


raw = open(paper_path, encoding="utf-8", errors="ignore").read()
cut = re.search(r"\n\s*(REFERENCES|References)\s*\n", raw)
body_raw = raw[: cut.start()] if cut else raw
sets = {}
for f in sorted(os.listdir(src_dir)):
    if f.endswith(".txt"):
        t = tokens(open(os.path.join(src_dir, f), encoding="utf-8", errors="ignore").read())
        sets[f[:-4]] = {tuple(t[i:i + N]) for i in range(len(t) - N + 1)}


def coverage(tok):
    cov_any = [False] * len(tok); per = defaultdict(lambda: [False] * len(tok))
    for i in range(len(tok) - N + 1):
        g = tuple(tok[i:i + N])
        for name, s in sets.items():
            if g in s:
                for j in range(i, i + N):
                    per[name][j] = True; cov_any[j] = True
    return cov_any, per


out = []
for label, text in (("whole paper (incl. references)", raw), ("body (references excluded)", body_raw)):
    tok = tokens(text); cov, per = coverage(tok)
    out.append(f"== {label}: {len(tok)} words, {N}-gram matches")
    out.append(f"   OVERALL similarity: {100 * sum(cov) / max(1, len(tok)):.2f}%")
    for name, c in sorted(per.items(), key=lambda kv: -sum(kv[1])):
        out.append(f"   {100 * sum(c) / len(tok):6.2f}%  {name}")
    if label.startswith("body"):
        spans, i = [], 0                                  # longest matched spans in the body
        while i < len(tok):
            if cov[i]:
                j = i
                while j < len(tok) and cov[j]:
                    j += 1
                spans.append((j - i, " ".join(tok[i:j])))
                i = j
            else:
                i += 1
        out.append("   longest matched passages:")
        for ln, s in sorted(spans, reverse=True)[:25]:
            out.append(f"   [{ln:3d}] {s[:220]}")
print("\n".join(out))
if report:
    open(report, "w").write("\n".join(out))
