"""Apply the rewrite lists of rewrite1.py and rewrite2.py to a file that lacks some passages (e.g. no appendix).
Entries found 0 times are reported and skipped; entries found a different non-zero number of times abort.
Usage: python apply_lenient.py in.tex out.tex
"""
import sys

t = open(sys.argv[1], encoding="utf-8").read()
for script in ("rewrite1.py", "rewrite2.py"):
    code = open(script, encoding="utf-8").read()
    head = code.split("\nsrc, dst")[0] if "\nsrc, dst" in code else code.split("\nt = open")[0]
    ns = {}
    exec(head, ns)
    applied = skipped = 0
    for old, new, n in ns["R"]:
        c = t.count(old)
        if c == 0:
            skipped += 1; print(f"  [{script}] not present: {old[:70]}"); continue
        if c != n and not (n == 2 and c == 1):
            sys.exit(f"count mismatch {c} vs {n}: {old[:70]}")
        t = t.replace(old, new); applied += 1
    print(f"{script}: applied {applied}, skipped {skipped}")
open(sys.argv[2], "w", encoding="utf-8").write(t)
