"""Round-2 rewording (small) and layout fix for the loss equation. Usage: python rewrite2.py in.tex out.tex"""
import sys

R = [
("Each model is fitted on insertion and tested on removal (I$\\to$R), and the roles are then swapped (R$\\to$I).",
 "Models fitted to insertion are evaluated on removal, written I$\\to$R; the opposite assignment is written R$\\to$I.", 1),
("These values are on par with the plain LAM (0.737 and 0.640, inside one standard deviation), but {\\method} additionally provides reverse actions in closed form.",
 "The plain LAM scores 0.737 and 0.640, within one standard deviation of these values, yet only {\\method} also supplies reverse actions in closed form.", 1),
("With $k=5$, the error reaches 1.008 (I$\\to$R) and 0.796 (R$\\to$I) without the anchor and 1.033 and 0.808 with it, compared with 1.099 and 0.872 for the plain LAM and 1.060 and 0.840 for sign reversal.",
 "For $k=5$, {\\method} without the anchor attains 1.008 on I$\\to$R and 0.796 on R$\\to$I, the anchored variant 1.033 and 0.808, the plain LAM 1.099 and 0.872, and sign reversal 1.060 and 0.840.", 1),
("Examples include additive composition~\\cite{wei2026aclam}, a consistency loss requiring the action inferred from a swapped frame pair to equal the negated forward action~\\cite{tang2026alam}, and group-action constraints on world models~\\cite{wang2026groupactions,keurti2023hae}.",
 "Examples are additive composition of latent actions~\\cite{wei2026aclam}, a consistency term asking that a swapped frame pair be explained by the negated forward action~\\cite{tang2026alam}, and group-action constraints imposed on world models~\\cite{wang2026groupactions,keurti2023hae}.", 1),
("The configuration of {\\method} used in the experiments, annotated with tensor sizes on the ALOHA toaster data: a $480\\times848$ image is mapped to a 128-D feature $o_t$, a 65-D lifted state $s_t$ and a 16-D latent action $z_t$, and a single $65\\times65$ operator $K(z_t)$ provides the forward step, the backward step and both reverse actions.",
 "Experimental configuration of {\\method} with tensor sizes for the ALOHA toaster data. Each camera image ($480\\times848$) is reduced to a feature $o_t\\in\\R^{128}$, lifted to $s_t\\in\\R^{65}$ and paired with a latent action $z_t\\in\\R^{16}$; the forward step, the backward step and both reverse actions all come from one operator $K(z_t)\\in\\R^{65\\times65}$.", 1),
("\\mathcal{L}&=\\ell_{\\mathrm{fwd}}+\\lambda_b\\,\\ell_{\\mathrm{bwd}}+\\lambda_r\\,(\\ell_{\\mathrm{rev}}+\\ell_{\\mathrm{neg}})+\\lambda_a\\,\\ell_{\\mathrm{idle}}+\\lambda_v\\,\\Omega+\\mathcal{L}_{\\mathrm{rec}},\\nonumber\\\\",
 "\\mathcal{L}&=\\ell_{\\mathrm{fwd}}+\\lambda_b\\,\\ell_{\\mathrm{bwd}}+\\lambda_r\\,(\\ell_{\\mathrm{rev}}+\\ell_{\\mathrm{neg}})\\nonumber\\\\\n&\\quad+\\lambda_a\\,\\ell_{\\mathrm{idle}}+\\lambda_v\\,\\Omega+\\mathcal{L}_{\\mathrm{rec}},\\nonumber\\\\", 1),
("\\ell_{\\mathrm{neg}}&=\\|z_t+\\zb_t\\|^2,\\quad \\ell_{\\mathrm{idle}}=\\textstyle\\sum_{t\\in\\mathcal{N}}\\|z_t\\|^2,\\quad \\Omega=\\textstyle\\sum_i\\|V_i\\|_F^2,",
 "\\ell_{\\mathrm{neg}}&=\\|z_t+\\zb_t\\|^2,\\quad \\ell_{\\mathrm{idle}}=\\textstyle\\sum_{t\\in\\mathcal{N}}\\|z_t\\|^2,\\nonumber\\\\\n\\Omega&=\\textstyle\\sum_i\\|V_i\\|_F^2,", 1),
]
t = open(sys.argv[1], encoding="utf-8").read()
bad = [(t.count(o), n, o[:80]) for o, _, n in R if t.count(o) != n]
if bad:
    print(*bad, sep="\n"); sys.exit(1)
for o, nw, _ in R:
    t = t.replace(o, nw)
open(sys.argv[2], "w", encoding="utf-8").write(t)
print("applied", len(R))
