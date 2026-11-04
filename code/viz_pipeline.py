"""Animate the RAVE pipeline on one held-out episode, showing every variable per step.

Input:  viz/export_<dir>_<fold>.npz written by `rave.py --export`, plus the raw HDF5 episode for camera frames.
Output: viz/rave_pipeline.mp4 (1920x1080).
"""
import os, sys
import h5py, cv2, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "viz", "export_I_0.npz")
E = np.load(src)
direction, ep = str(E["direction"]), int(E["episode_id"])
folder = {"I": "toaster_forward", "R": "toaster_backward"}[direction]
task = {"I": "insertion", "R": "removal"}[direction]
T = len(E["z"])                                   # transitions
RAW_PER_STEP = 5 * int(E["stride"])               # 10 Hz features -> 50 Hz raw steps

with h5py.File(f"/home/pemaase/patrick/{folder}/episode_{ep}.hdf5") as f:
    frames = [cv2.imdecode(f["observations/images/cam_high"][t * RAW_PER_STEP], 1)[:, :, ::-1] for t in range(T + 1)]

# releases: gripper opens past 0.9 after holding an object (0.05 < g < 0.9) for >= 1 s
q = E["qpos"]; g = q[:, [6, 13]]; hold = max(1, 10 // int(E["stride"])); rel = []
for a in range(2):
    run = 0
    for t in range(1, len(g)):
        run = run + 1 if 0.05 < g[t - 1, a] < 0.9 else 0
        if run >= hold and g[t, a] >= 0.9:
            rel.append(t - 1)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DIV = LinearSegmentedColormap.from_list("div", ["#c2491f", "#f3b39a", "#f0efec", "#86b6ef", "#184f95"])
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})

o, s, z, zbar, zu = E["o"], E["s"], E["z"], E["zbar"], E["z_undo"]
K, G0, kap, fe, be = E["K"], E["G0"], E["kappa"], E["fwd_err"], E["bwd_err"]
d1 = s.shape[1]; m = z.shape[1]
lim_o = np.abs(o).max() * 1.05; lim_s = np.abs(s[:, :-1]).max() * 1.05
lim_z = max(np.abs(z).max(), np.abs(zbar).max(), np.abs(zu).max()) * 1.1
KmI = K - np.eye(d1)[None]; lim_k = np.percentile(np.abs(KmI), 99.5)
lim_g = np.percentile(np.abs(G0), 99.5)

fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
gs = fig.add_gridspec(3, 3, left=0.045, right=0.955, top=0.875, bottom=0.06, hspace=0.62, wspace=0.26)
title = fig.text(0.04, 0.955, "", fontsize=20, weight="bold")
fig.text(0.04, 0.92, f"Trained on {task} (fold 0), held-out episode {ep}.  One step = {0.1 * int(E['stride']):.1f} s.  "
         f"Drift residual = {float(E['drift_residual']):.3f}", fontsize=12, color=INK2)
for y, lbl in ((0.87, "INPUT"), (0.585, "LATENT"), (0.30, "OUTPUT")):
    fig.text(0.006, y, lbl, rotation=90, fontsize=13, weight="bold", color=INK2, va="top")

ax_img = fig.add_subplot(gs[0, 0]); ax_o = fig.add_subplot(gs[0, 1]); ax_s = fig.add_subplot(gs[0, 2])
ax_z = fig.add_subplot(gs[1, 0]); ax_k = fig.add_subplot(gs[1, 1]); ax_g = fig.add_subplot(gs[1, 2])
ax_rev = fig.add_subplot(gs[2, 0]); ax_err = fig.add_subplot(gs[2, 1]); ax_kap = fig.add_subplot(gs[2, 2])

# --- static parts
img = ax_img.imshow(frames[0]); ax_img.set_xticks([]); ax_img.set_yticks([])
ax_img.set_title("1. Camera frame  $o_t$   (480 × 848 × 3)", loc="left", fontsize=13)
bo = ax_o.bar(np.arange(o.shape[1]), o[0], color=BLUE, width=0.9)
ax_o.set_ylim(-lim_o, lim_o); ax_o.set_xlim(-1, o.shape[1])
ax_o.set_title("2. DINOv2 + PCA feature  $o_t \\in \\mathbb{R}^{128}$", loc="left", fontsize=13)
ax_o.set_xlabel("feature index")
bs = ax_s.bar(np.arange(d1 - 1), s[0, :-1], color=BLUE, width=0.85)
ax_s.set_ylim(-lim_s, lim_s); ax_s.set_xlim(-1, d1)
ax_s.set_title("3. Encoder $\phi$: lifted state  $s_t = [\\phi(o_t); 1] \\in \\mathbb{R}^{65}$", loc="left", fontsize=13)
ax_s.set_xlabel("latent index (constant coordinate = 1 not shown)")

x = np.arange(m)
bz = ax_z.bar(x, z[0], color=BLUE, width=0.7)
ax_z.set_ylim(-lim_z, lim_z); ax_z.set_xticks(x); ax_z.axhline(0, color=GRID, lw=1)
ax_z.set_title("4. Inverse dynamics: latent action  $z_t = I_\\psi(o_t, o_{t+1}) \\in \\mathbb{R}^{16}$", loc="left", fontsize=13)
ax_z.set_xlabel("action dimension i")
hk = ax_k.imshow(KmI[0], cmap=DIV, norm=TwoSlopeNorm(0, -lim_k, lim_k), aspect="auto")
ax_k.set_title("5. Koopman step  $K(z_t) - I \\in \\mathbb{R}^{65\\times65}$", loc="left", fontsize=13)
ax_k.set_xticks([]); ax_k.set_yticks([])
fig.colorbar(hk, ax=ax_k, fraction=0.046, pad=0.02)
hg = ax_g.imshow(G0, cmap=DIV, norm=TwoSlopeNorm(0, -lim_g, lim_g), aspect="auto")
ax_g.set_title("Drift generator  $G_0$  (fixed after training)", loc="left", fontsize=13)
ax_g.set_xticks([]); ax_g.set_yticks([])
fig.colorbar(hg, ax=ax_g, fraction=0.046, pad=0.02)

w = 0.27
b1 = ax_rev.bar(x - w, -z[0], w, color=BLUE, label="$-z_t$")
b2 = ax_rev.bar(x, zbar[0], w, color=ORANGE, label="$\\bar z_t = I_\\psi(o_{t+1}, o_t)$")
b3 = ax_rev.bar(x + w, zu[0], w, color=AQUA, label="$z_{undo} = -z_t - 2c$")
ax_rev.set_ylim(-lim_z, lim_z); ax_rev.set_xticks(x); ax_rev.axhline(0, color=GRID, lw=1)
ax_rev.legend(ncol=3, fontsize=10, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.02))
ax_rev.set_title("6. Reverse actions  $\\in \\mathbb{R}^{16}$", loc="left", fontsize=13, pad=22)
rev_txt = ax_rev.text(0.99, 0.04, "", transform=ax_rev.transAxes, ha="right", fontsize=11, color=INK2)

ts = np.arange(T)
ax_err.plot(ts, fe, color=BLUE, lw=1.2, alpha=0.35); ax_err.plot(ts, be, color=ORANGE, lw=1.2, alpha=0.35)
le1, = ax_err.plot([], [], color=BLUE, lw=2.2, label="forward  $\\hat o_{t+1}=D(e^{A}s_t)$")
le2, = ax_err.plot([], [], color=ORANGE, lw=2.2, label="backward  $\\hat o_t=D(e^{-A}s_{t+1})$")
ve = ax_err.axvline(0, color=INK2, lw=1, ls=":")
ax_err.axhline(1.0, color=INK2, lw=0.8, ls="--"); ax_err.text(T - 1, 1.0, "predict no change", fontsize=9,
                                                            color=INK2, ha="right", va="bottom")
ax_err.set_xlim(0, T - 1); ax_err.set_ylim(0, max(1.15, np.r_[fe, be].max() * 1.08))
ax_err.set_title("7. Prediction error per step (relative to no change)", loc="left", fontsize=13)
ax_err.set_xlabel("step t"); ax_err.legend(fontsize=10, frameon=False, loc="upper left")

ax_kap.plot(ts, kap, color=YELLOW, lw=1.2, alpha=0.35)
lk, = ax_kap.plot([], [], color=YELLOW, lw=2.2)
for r in rel:
    if r < T:
        ax_kap.axvspan(r - 0.5, r + 0.5, color=ORANGE, alpha=0.18, lw=0)
vk = ax_kap.axvline(0, color=INK2, lw=1, ls=":")
ax_kap.set_xlim(0, T - 1); ax_kap.set_ylim(min(kap) * 0.98, max(kap) * 1.02)
ax_kap.set_title("8. Reversibility score  $\\kappa_t = \\mathrm{cond}\\,K(z_t)$", loc="left", fontsize=13)
ax_kap.set_xlabel("step t   (shaded: gripper release)")
kap_txt = ax_kap.text(0.99, 0.92, "", transform=ax_kap.transAxes, ha="right", fontsize=11, color=INK2)

# --- flow arrows between stages (figure coordinates)
fig.canvas.draw()
def arrow(p, q):
    fig.patches.append(matplotlib.patches.FancyArrowPatch(p, q, transform=fig.transFigure, arrowstyle="-|>",
                       mutation_scale=18, color=INK2, lw=1.4))
P = {n: a.get_position() for n, a in dict(img=ax_img, o=ax_o, s=ax_s, z=ax_z, k=ax_k, g=ax_g,
                                         rev=ax_rev, err=ax_err, kap=ax_kap).items()}
mid = lambda b: (b.y0 + b.y1) / 2
cx = lambda b: (b.x0 + b.x1) / 2
arrow((P["img"].x1 + 0.005, mid(P["img"])), (P["o"].x0 - 0.012, mid(P["o"])))      # frame -> feature
arrow((P["o"].x1 + 0.005, mid(P["o"])), (P["s"].x0 - 0.012, mid(P["s"])))          # feature -> state
arrow((P["z"].x1 + 0.005, mid(P["z"])), (P["k"].x0 - 0.008, mid(P["k"])))          # z -> K(z)
arrow((P["g"].x0 - 0.008, mid(P["g"])), (P["k"].x1 + 0.035, mid(P["k"])))          # G0 -> K(z)
arrow((cx(P["s"]), P["s"].y0 - 0.045), (cx(P["k"]) + 0.06, P["k"].y1 + 0.03))      # s_t -> Koopman step
arrow((cx(P["z"]), P["z"].y0 - 0.05), (cx(P["rev"]), P["rev"].y1 + 0.045))          # z -> reverse actions
arrow((cx(P["k"]), P["k"].y0 - 0.005), (cx(P["err"]), P["err"].y1 + 0.03))          # K -> predictions
arrow((P["k"].x1 + 0.03, P["k"].y0 - 0.005), (cx(P["kap"]), P["kap"].y1 + 0.03))    # K -> kappa


def update(t):
    title.set_text(f"RAVE pipeline  ·  step t = {t} / {T - 1}   ({t * 0.1 * int(E['stride']):.1f} s)")
    img.set_data(frames[t])
    for b, v in zip(bo, o[t]): b.set_height(v)
    for b, v in zip(bs, s[t, :-1]): b.set_height(v)
    for b, v in zip(bz, z[t]): b.set_height(v)
    hk.set_data(KmI[t])
    for bars, vals in ((b1, -z[t]), (b2, zbar[t]), (b3, zu[t])):
        for b, v in zip(bars, vals): b.set_height(v)
    cos = float(np.dot(zbar[t], -z[t]) / (np.linalg.norm(zbar[t]) * np.linalg.norm(z[t]) + 1e-8))
    rev_txt.set_text(f"cos($\\bar z_t$, $-z_t$) = {cos:+.2f}")
    le1.set_data(ts[: t + 1], fe[: t + 1]); le2.set_data(ts[: t + 1], be[: t + 1]); ve.set_xdata([t, t])
    lk.set_data(ts[: t + 1], kap[: t + 1]); vk.set_xdata([t, t])
    kap_txt.set_text(f"$\\kappa_t$ = {kap[t]:.3f}" + ("   release" if t in rel else ""))
    return []


anim = FuncAnimation(fig, update, frames=T, blit=False)
out = os.path.join(ROOT, "viz", "rave_pipeline.mp4")
anim.save(out, writer=FFMpegWriter(fps=2, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]))
update(min(rel[0] if rel else T // 2, T - 1)); fig.savefig(os.path.join(ROOT, "viz", "rave_pipeline_still.png"), dpi=80)
print("saved", out, "steps", T, "releases", rel)
