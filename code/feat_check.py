"""Feature check: can a linear probe decode the scene stage (S0 / S1 / S2) from each feature variant?

Stage labels come from gripper release events (grasp -> open): insertion goes S0 -> S1 -> S2 at its two
releases, removal goes S2 -> S1 -> S0. Episodes without exactly two releases are skipped.
"""
import glob, json, h5py, cv2, numpy as np, torch, torch.nn.functional as F

dev = "cuda"
torch.manual_seed(0)
model = torch.hub.load("/home/pemaase/.cache/torch/hub/facebookresearch_dinov2_main", "dinov2_vits14",
                       source="local").to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
SUB, N_EP = 25, 20                                   # 2 Hz frames, 20 episodes per direction


@torch.no_grad()
def dino(frames):
    x = torch.from_numpy(np.stack([f[:, :, ::-1] for f in frames]).copy()).to(dev).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
    out = model.forward_features((x - mean) / std)
    cls, pt = out["x_norm_clstoken"], out["x_norm_patchtokens"].view(-1, 16, 28, 384).permute(0, 3, 1, 2)
    g24 = F.adaptive_avg_pool2d(pt, (2, 4)).flatten(1)
    g47 = F.adaptive_avg_pool2d(pt, (4, 7)).flatten(1)
    return cls.cpu().numpy(), g24.cpu().numpy(), g47.cpu().numpy()


def releases(q, min_hold):
    """Gripper opens past 0.9 after holding an object (0.05 < g < 0.9) for at least min_hold steps."""
    out = []
    for a in (6, 13):
        g, run = q[:, a], 0
        for t in range(1, len(g)):
            run = run + 1 if 0.05 < g[t - 1] < 0.9 else 0
            if run >= min_hold and g[t] >= 0.9:
                out.append(t)
    return sorted(out)


def stages(q, direction):
    rel = releases(q, 50)                            # 1 s hold at 50 Hz
    if len(rel) != 2:
        return None
    s = np.zeros(len(q), int); s[rel[0]:] = 1; s[rel[1]:] = 2
    return s if direction == "I" else 2 - s


feats, labels, groups = {}, [], []
n_skip = {"I": 0, "R": 0}
for direction, d in (("I", "toaster_forward"), ("R", "toaster_backward")):
    files = sorted(glob.glob(f"/home/pemaase/patrick/{d}/*.hdf5"), key=lambda f: int(f.split("_")[-1][:-5]))
    used = 0
    for p in files:
        if used == N_EP:
            break
        with h5py.File(p) as f:
            q = f["observations/qpos"][:]
            s = stages(q, direction)
            if s is None:
                n_skip[direction] += 1; continue
            idx = np.arange(0, len(q), SUB)
            out = {}
            for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                out[cam] = dino([cv2.imdecode(f[f"observations/images/{cam}"][t], 1) for t in idx])
        top, lw, rw = out["cam_high"], out["cam_left_wrist"], out["cam_right_wrist"]
        variants = {
            "top: CLS + 2x4 grid (current)": np.c_[top[0], top[1]],
            "top: CLS + 4x7 grid": np.c_[top[0], top[2]],
            "wrists: CLS + 2x4 grid (L+R)": np.c_[lw[0], lw[1], rw[0], rw[1]],
            "top 4x7 + wrists 2x4": np.c_[top[0], top[2], lw[0], lw[1], rw[0], rw[1]],
        }
        for k, v in variants.items():
            feats.setdefault(k, []).append(v)
        labels.append(s[idx]); groups.append(np.full(len(idx), len(groups)))
        used += 1
        print(direction, p.split("/")[-1], flush=True)

y = np.concatenate(labels); grp = np.concatenate(groups); n_ep = len(labels)
print("episodes used", n_ep, "skipped (not exactly 2 releases)", n_skip, "stage counts", np.bincount(y))


def probe(X):
    """5-fold episode-split logistic regression on 256 PCA dims; balanced accuracy."""
    accs = []
    for k in range(5):
        te = np.isin(grp, np.arange(k, n_ep, 5)); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs = (X - mu) / sd
        U = np.linalg.svd(Xs[tr][::2], full_matrices=False)[2][:256].T
        Z = torch.tensor(Xs @ U, dtype=torch.float32, device=dev); Y = torch.tensor(y, device=dev)
        W = torch.zeros(Z.shape[1], 3, device=dev, requires_grad=True); b = torch.zeros(3, device=dev, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=1e-2)
        trm = torch.tensor(tr, device=dev)
        for _ in range(500):
            loss = F.cross_entropy(Z[trm] @ W + b, Y[trm]) + 1e-3 * W.pow(2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        pred = (Z[~trm] @ W + b).argmax(1).cpu().numpy(); yt = y[te]
        accs.append(np.mean([np.mean(pred[yt == c] == c) for c in range(3) if (yt == c).any()]))
    return float(np.mean(accs)), float(np.std(accs))


res = {}
for k, v in feats.items():
    res[k] = probe(np.concatenate(v))
    print(f"{k:34s} balanced acc {res[k][0]:.3f} ± {res[k][1]:.3f}  (chance 0.333)", flush=True)
json.dump(dict(results=res, n_episodes=n_ep, skipped=n_skip, stage_counts=np.bincount(y).tolist()),
          open("/home/pemaase/patrick/rave/feat_check.json", "w"), indent=1)
