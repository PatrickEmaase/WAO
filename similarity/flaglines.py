"""Flag LaTeX source lines whose words fall inside 8-grams shared with the comparison corpus.

Usage: python flaglines.py paper.tex sources_dir [min_cov] > flags.txt
Each output row: line number, coverage of that line, region (prose / float), the line text.
"""
import os, re, sys

tex_path, src_dir = sys.argv[1], sys.argv[2]
MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
N = 8


def toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def clean(line):
    line = re.sub(r"(?<!\\)%.*", "", line)
    line = line.replace("{\\method}", "RAVI").replace("\\method{}", "RAVI").replace("\\method", "RAVI")
    line = re.sub(r"\\(cite|ref|label|eqref)\{[^}]*\}", " ", line)
    line = re.sub(r"\$[^$]*\$", " ", line)                     # drop inline math
    line = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", line)    # drop command names, keep their arguments
    return line


src = set()
for f in os.listdir(src_dir):
    if f.endswith(".txt"):
        t = toks(open(os.path.join(src_dir, f), encoding="utf-8", errors="ignore").read())
        src |= {tuple(t[i:i + N]) for i in range(len(t) - N + 1)}

lines = open(tex_path, encoding="utf-8").read().split("\n")
stream, owner, region, depth = [], [], [], 0
for k, ln in enumerate(lines):
    if re.search(r"\\begin\{(figure|table|tikzpicture|tabular|algorithm)\*?\}", ln):
        depth += 1
    reg = "float" if depth > 0 or "\\caption" in ln else "prose"
    if re.search(r"\\end\{(figure|table|tikzpicture|tabular|algorithm)\*?\}", ln):
        depth = max(0, depth - 1)
    region.append(reg)
    for w in toks(clean(ln)):
        stream.append(w); owner.append(k)

cov = [False] * len(stream)
for i in range(len(stream) - N + 1):
    if tuple(stream[i:i + N]) in src:
        for j in range(i, i + N):
            cov[j] = True

tot = {"prose": [0, 0], "float": [0, 0]}
per = {}
for w, k, c in zip(stream, owner, cov):
    per.setdefault(k, [0, 0]); per[k][0] += 1; per[k][1] += c
    tot[region[k]][0] += 1; tot[region[k]][1] += c
for reg, (n, c) in tot.items():
    print(f"# {reg}: {n} words, {100 * c / max(1, n):.2f}% covered")
for k in sorted(per):
    n, c = per[k]
    if n and c / n > MIN and c > 0:
        print(f"{k + 1}\t{100 * c / n:5.1f}\t{region[k]}\t{lines[k].strip()[:400]}")
