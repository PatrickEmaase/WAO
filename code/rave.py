"""RAVE: Reverse Action via Koopman Operator -- training and evaluation on the ALOHA toaster pair.

Directions: I = insertion (toaster_forward), R = removal (toaster_backward).
Usage: python rave.py --train I --fold 0 --model rave      (models: lam, signrev, rave_noanchor, rave, oracle)
"""
import argparse, copy, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--train", choices=["I", "R"], required=True)
p.add_argument("--fold", type=int, required=True)
p.add_argument("--model", required=True)
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--pca", type=int, default=128)
p.add_argument("--d", type=int, default=64)
p.add_argument("--m", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--stride", type=int, default=5)           # 10 Hz features -> one step = stride / 10 s
p.add_argument("--sym_reg", type=float, default=1e-3)     # L2 on the symmetric part of the action generators
p.add_argument("--export", action="store_true")           # dump all variables for one held-out episode (viz/)
p.add_argument("--out", default="results")
args = p.parse_args()
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda"
ROOT = os.path.dirname(os.path.abspath(__file__))
FILES = {"I": "toaster_forward.npz", "R": "toaster_backward.npz"}
NFOLD, IDLE_Q = 5, 0.15                      # robot-idle anchor = slowest 15% of 10 Hz steps


# ---------------------------------------------------------------- data
def load(dirn):
    """Return the 10 Hz arrays plus `stride` phase-shifted copies at the coarse step (offset 0 = evaluation)."""
    z = dict(np.load(os.path.join(ROOT, FILES[dirn])))
    moving = np.diff(z["qpos"], axis=1).var(1).sum(1) > 1e-8                   # drop static recordings
    z = {k: v[moving] for k, v in z.items()}
    x = np.concatenate([z["cls"], z["grid"].astype(np.float32)], -1)          # (E, 250, 3456)
    S = args.stride; n = (x.shape[1] // S) * S
    sp = z["speed"][:, :n].reshape(len(x), -1, S).mean(-1)                    # mean joint speed over each step
    return dict(x=x[:, :n:S], qpos=z["qpos"][:, :n:S], speed=sp, x_all=x[:, :n], speed_all=z["speed"][:, :n],
                ids=z["ids"])


def folds(n, k):
    idx = np.random.RandomState(123).permutation(n)
    return np.sort(idx[k::NFOLD])


A, B = args.train, ("R" if args.train == "I" else "I")
DA, DB = load(A), load(B)
teA, teB = folds(len(DA["x"]), args.fold), folds(len(DB["x"]), args.fold)
trA = np.setdiff1d(np.arange(len(DA["x"])), teA)
trB = np.setdiff1d(np.arange(len(DB["x"])), teB)

fit = DA["x"][trA].reshape(-1, DA["x"].shape[-1])
if args.model == "oracle":
    fit = np.concatenate([fit, DB["x"][trB].reshape(-1, DB["x"].shape[-1])])
mu, sd = fit.mean(0), fit.std(0) + 1e-6
U = np.linalg.svd((fit[::4] - mu) / sd, full_matrices=False)[2][: args.pca].T
scale = np.sqrt(((((fit[::4] - mu) / sd) @ U) ** 2).mean())                 # unit average variance per component
proj = lambda x: ((((x - mu) / sd) @ U) / scale).astype(np.float32)
for D in (DA, DB):
    D["o"] = proj(D["x"].reshape(-1, D["x"].shape[-1])).reshape(*D["x"].shape[:2], -1)
    D["o_all"] = proj(D["x_all"].reshape(-1, D["x_all"].shape[-1])).reshape(*D["x_all"].shape[:2], -1)
    D["dq"] = np.diff(D["qpos"], axis=1)                                      # (E, T-1, 14)
thr = np.quantile(DA["speed"][trA], IDLE_Q)


def pairs(D, eps, all_phases=False):
    """Transitions one coarse step apart; with all_phases, every 10 Hz start offset (training augmentation)."""
    S = args.stride if all_phases else 1
    o = torch.tensor(D["o_all"][eps] if all_phases else D["o"][eps], device=dev)
    sp = D["speed_all"][eps] if all_phases else D["speed"][eps]
    if all_phases:                                                            # mean speed over each S-frame window
        sp = np.stack([np.convolve(r, np.ones(S) / S, mode="full")[S - 1: S - 1 + len(r)] for r in sp])
    idle = torch.tensor(sp[:, :-S] < thr, device=dev)
    return o[:, :-S].reshape(-1, o.shape[-1]), o[:, S:].reshape(-1, o.shape[-1]), idle.reshape(-1)


train_sets = [pairs(DA, trA, True)] + ([pairs(DB, trB, True)] if args.model == "oracle" else [])
O0 = torch.cat([s[0] for s in train_sets]); O1 = torch.cat([s[1] for s in train_sets])
IDLE = torch.cat([s[2] for s in train_sets])

# ---------------------------------------------------------------- models
P, d, m = args.pca, args.d, args.m
mlp = lambda i, o: nn.Sequential(nn.Linear(i, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, o))


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec, self.idm_net = mlp(P, d), mlp(d, P), mlp(3 * P, m)

    def idm(self, a, b):
        return self.idm_net(torch.cat([a, b, b - a], -1))


class LAM(Base):                                  # MLP forward model; reverse handled per variant
    def __init__(self):
        super().__init__()
        self.fdm, self.bdm = mlp(d + m, d), mlp(d + m, d)

    def encode(self, o): return self.enc(o)
    def decode(self, s): return self.dec(s)
    def fwd(self, s, z): return s + self.fdm(torch.cat([s, z], -1))
    def bwd(self, s, z): return s + self.bdm(torch.cat([s, z], -1))


class RAVE(Base):                                 # K(z) = exp(G0 + sum z_i G_i) acting on [s; 1]
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(0.01 * torch.randn(m, d, d)); self.b = nn.Parameter(0.01 * torch.randn(m, d))
        self.V = nn.Parameter(torch.zeros(m, d, d))                                    # symmetric part, L2-penalized
        self.S0 = nn.Parameter(0.01 * torch.randn(d, d)); self.L0 = nn.Parameter(0.01 * torch.randn(d, d))
        self.b0 = nn.Parameter(torch.zeros(d))

    def encode(self, o):
        s = self.enc(o)
        return torch.cat([s, torch.ones_like(s[:, :1])], -1)

    def decode(self, s): return self.dec(s[:, :-1])

    def gens(self):
        Gi = torch.zeros(m, d + 1, d + 1, device=self.W.device)
        Gi[:, :d, :d] = (self.W - self.W.transpose(1, 2)) + 0.5 * (self.V + self.V.transpose(1, 2))
        Gi[:, :d, d] = self.b                                                          # + translation column
        G0 = torch.zeros(d + 1, d + 1, device=self.W.device)
        G0[:d, :d] = (self.S0 - self.S0.T) - self.L0 @ self.L0.T; G0[:d, d] = self.b0  # skew + NSD drift
        return G0, Gi

    def gen(self, z, drift=1.0):
        G0, Gi = self.gens()
        return drift * G0 + torch.einsum("bm,mij->bij", z, Gi)

    @staticmethod
    def step(A, s): return (torch.matrix_exp(A) @ s.unsqueeze(-1)).squeeze(-1)

    def fwd(self, s, z): return self.step(self.gen(z), s)

    def drift_action(self):
        """c minimising E_s ||(G0 - sum_i c_i G_i) s||^2 over visited latents S_ref, plus the relative residual."""
        G0, Gi = self.gens(); S = self.S_ref                                           # (N, d+1)
        X = torch.einsum("mij,nj->nim", Gi, S).reshape(-1, Gi.shape[0])                 # (N(d+1), m)
        y = (S @ G0.T).reshape(-1, 1)
        c = torch.linalg.lstsq(X, y).solution.squeeze(-1)
        return c, ((y.squeeze(-1) - X @ c).norm() / y.norm()).item()


kind = args.model
model = (LAM() if kind in ("lam", "signrev") else RAVE()).to(dev)
mse = F.mse_loss


def losses(o0, o1, idle):
    s0, s1 = model.encode(o0), model.encode(o1)
    z = model.idm(o0, o1); z_n = z + 0.05 * torch.randn_like(z)             # small bottleneck noise
    s1h = model.fwd(s0, z_n)
    L = (mse(model.decode(s0), o0) + mse(model.decode(s1), o1) + mse(model.decode(s1h), o1)
         + mse(s1h, s1) + 1e-3 * z.pow(2).mean())
    zb = model.idm(o1, o0)                                                   # IDM on the swapped pair
    if kind == "lam":                                                        # separate backward model, reversed clips
        L = L + mse(model.decode(model.bwd(s1, zb)), o0)
    elif kind == "signrev":                                                  # shared forward model, z_bar = -z
        L = L + mse(model.decode(model.fwd(s1, zb)), o0) + mse(zb, -z)
    else:                                                                    # exact inverse + drift-aware reversal
        L = L + mse(model.decode(model.step(-model.gen(z_n), s1)), o0)
        L = L + mse(model.decode(model.step(model.gen(zb, drift=-1.0), s1)), o0) + mse(zb, -z)
        if kind in ("rave", "oracle") and idle.any():
            L = L + z[idle].pow(2).mean()
        L = L + args.sym_reg * model.V.pow(2).sum()
    return L


opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for it in range(args.steps):
    i = torch.randint(0, len(O0), (512,), device=dev)
    loss = losses(O0[i], O1[i], IDLE[i])
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if it % 1000 == 0:
        print(f"it {it} loss {loss.item():.4f}", flush=True)
model.eval()
if isinstance(model, RAVE):
    with torch.no_grad():                                                     # visited latents for the drift fit
        model.S_ref = model.encode(O0[torch.randperm(len(O0), device=dev)[:2000]])

if args.export and isinstance(model, RAVE):
    with torch.no_grad():                                                     # every pipeline variable, one test episode
        e = teA[0]; o = torch.tensor(DA["o"][e], device=dev); s = model.encode(o)
        z = model.idm(o[:-1], o[1:]); zbar = model.idm(o[1:], o[:-1]); c, resid = model.drift_action()
        A_ = model.gen(z); K = torch.matrix_exp(A_); G0, Gi = model.gens()
        fwd = model.decode(model.step(A_, s[:-1])); bwd = model.decode(model.step(-A_, s[1:]))
        base = (o[1:] - o[:-1]).pow(2).mean(1)
        sv = torch.linalg.svdvals(K)
        os.makedirs(os.path.join(ROOT, "viz"), exist_ok=True)
        np.savez(os.path.join(ROOT, "viz", f"export_{A}_{args.fold}.npz"),
                 episode_id=int(DA["ids"][e]), direction=A, stride=args.stride,
                 o=o.cpu().numpy(), s=s.cpu().numpy(), z=z.cpu().numpy(), zbar=zbar.cpu().numpy(),
                 z_undo=(-z - 2 * c).cpu().numpy(), c=c.cpu().numpy(), drift_residual=resid,
                 K=K.cpu().numpy(), G0=G0.cpu().numpy(), Gi=Gi.cpu().numpy(),
                 kappa=(sv[:, 0] / sv[:, -1]).cpu().numpy(),
                 fwd_err=((fwd - o[1:]).pow(2).mean(1) / base).cpu().numpy(),
                 bwd_err=((bwd - o[:-1]).pow(2).mean(1) / base).cpu().numpy(),
                 qpos=DA["qpos"][e])
    print("exported episode", int(DA["ids"][e]), flush=True)

# ---------------------------------------------------------------- evaluation helpers
T = lambda a: torch.tensor(np.ascontiguousarray(a), device=dev)


def ep_actions(D, e):
    o = T(D["o"][e])
    return o, model.idm(o[:-1], o[1:])


def reverse_actions(o, z):
    """Actions that undo each forward transition, in reversed order, under the model's FORWARD dynamics."""
    if kind == "lam":
        zr = model.idm(o[1:], o[:-1])
    elif kind == "signrev":
        zr = -z
    else:
        c, _ = model.drift_action(); zr = -z - 2 * c
    return zr.flip(0)


def rollout(o_start, zs):
    s = model.encode(o_start[None]); out = [s]
    for z in zs:
        s = model.fwd(s, z[None]); out.append(s)
    return model.decode(torch.cat(out))


def dtw_path(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    C = 1 - a @ b.T; n, k = C.shape
    acc = np.full((n + 1, k + 1), np.inf); acc[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            acc[i, j] = C[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, path = n, k, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        i, j = (i - 1, j - 1) if step == 0 else (i - 1, j) if step == 1 else (i, j - 1)
    return path[::-1]


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


res = dict(train=A, fold=args.fold, model=kind)
with torch.no_grad():
    # 1. backward (and forward) prediction on held-out episodes of the training direction
    o0, o1, _ = pairs(DA, teA); s1 = model.encode(o1); z = model.idm(o0, o1)
    if kind == "lam":
        ob = model.decode(model.bwd(s1, model.idm(o1, o0)))
    elif kind == "signrev":
        ob = model.decode(model.fwd(s1, -z))
    else:
        ob = model.decode(model.step(-model.gen(z), s1))
    res["backward_rel_err"] = (mse(ob, o0) / mse(o1, o0)).item()
    res["forward_rel_err"] = (mse(model.decode(model.fwd(model.encode(o0), z)), o1) / mse(o0, o1)).item()
    if kind not in ("lam", "signrev"):
        res["drift_residual"] = model.drift_action()[1]

    # stage prototypes (evaluation only): S0 = slices on plate, S2 = slices in toaster
    DI, DR = (DA, DB) if A == "I" else (DB, DA)
    S0 = np.concatenate([DI["o"][:, 0], DR["o"][:, -1]]).mean(0)
    S2 = np.concatenate([DI["o"][:, -1], DR["o"][:, 0]]).mean(0)
    tgt, src = (S0, S2) if A == "I" else (S2, S0)                            # the reverse of A ends where A starts

    # 2-4. transfer to the opposite direction; each B test episode is paired with an A test episode
    reach, dist, cos_al, probe_err, naive_err = [], [], [], [], []
    Ztr = torch.cat([ep_actions(DA, e)[1] for e in trA]).cpu().numpy(); DQtr = DA["dq"][trA].reshape(-1, 14)
    X = np.c_[Ztr, np.ones(len(Ztr))]
    Wp = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ DQtr)
    for j, eb in enumerate(teB):
        ea = teA[j % len(teA)]
        oa, za = ep_actions(DA, ea); ob_, zb = ep_actions(DB, eb)
        zr = zb if kind == "oracle" else reverse_actions(oa, za)
        end = rollout(ob_[0], zr).cpu().numpy()[-1]
        reach.append(float(np.linalg.norm(end - tgt) < np.linalg.norm(end - src)))
        dist.append(np.linalg.norm(end - tgt) / np.linalg.norm(DB["o"][eb][0] - tgt))
        path = dtw_path(DA["o"][ea][::-1], DB["o"][eb])                      # reversed A frames vs real B frames
        n = len(zr); zr_np, zb_np = zr.cpu().numpy(), zb.cpu().numpy()
        dq_naive = -DA["dq"][ea][::-1]                                       # proprioceptive negation reference
        pr = np.c_[zr_np, np.ones(n)] @ Wp
        var = DB["dq"][teB].reshape(-1, 14).var(0).sum()                    # pooled over test episodes
        al = [(k, l) for k, l in path if k < n and l < n]
        ka, la = np.array([a for a, _ in al]), np.array([b for _, b in al])
        cos_al.append(np.mean(np.sum(zr_np[ka] * zb_np[la], 1)
                              / (np.linalg.norm(zr_np[ka], axis=1) * np.linalg.norm(zb_np[la], axis=1) + 1e-8)))
        probe_err.append(((pr[ka] - DB["dq"][eb][la]) ** 2).sum(1).mean() / var)
        naive_err.append(((dq_naive[ka] - DB["dq"][eb][la]) ** 2).sum(1).mean() / var)
    res.update(stage_reach=float(np.mean(reach)), end_dist_ratio=float(np.mean(dist)),
               action_cos=float(np.mean(cos_al)), joint_probe_nmse=float(np.mean(probe_err)),
               joint_negation_nmse=float(np.mean(naive_err)))


# 5. reversibility detection on held-out A episodes: post-release frames vs grasped-carry frames
def cond_scores(o):
    if kind in ("lam", "signrev"):
        s = model.encode(o[:-1]).detach(); z = model.idm(o[:-1], o[1:]).detach()
        J = torch.func.vmap(torch.func.jacrev(lambda x, zz: model.fwd(x[None], zz[None])[0]))(s, z)
        sv = torch.linalg.svdvals(J)
        return (sv[:, 0] / sv[:, -1]).detach().cpu().numpy()
    with torch.no_grad():
        sv = torch.linalg.svdvals(torch.matrix_exp(model.gen(model.idm(o[:-1], o[1:]))))
        return (sv[:, 0] / sv[:, -1]).cpu().numpy()


pos, neg = [], []
for e in teA:
    g = DA["qpos"][e][:, [6, 13]]; kap = cond_scores(T(DA["o"][e])); n = len(kap)
    grasp = (g > 0.05) & (g < 0.9)
    rel = np.zeros(n, bool); hold = max(1, 10 // args.stride)                 # release = open after a 1 s grasp
    for a in range(2):
        run = 0
        for t in range(1, n):
            run = run + 1 if grasp[t - 1, a] else 0
            if run >= hold and g[t, a] >= 0.9:
                rel[t:t + max(1, 5 // args.stride)] = True                   # the 0.5 s after a release
    carry = grasp[:n].any(1) & (DA["speed"][e][:n] > thr) & ~rel
    pos += list(kap[rel]); neg += list(kap[carry])
res["reversibility_auroc"] = float(auroc(np.array(pos), np.array(neg))) if pos and neg else None
res["n_release_frames"], res["n_carry_frames"] = len(pos), len(neg)

# 6. few-shot: adapt only the IDM on k episodes of B; forward error on B test episodes
if kind != "oracle":
    base = copy.deepcopy(model.state_dict()); o0b, o1b, _ = pairs(DB, teB); res["fewshot_fwd_rel_err"] = {}
    for k in (0, 1, 3, 5):
        model.load_state_dict(base)
        if k:
            a0, a1, _ = pairs(DB, trB[:k]); opt = torch.optim.Adam(model.idm_net.parameters(), lr=1e-3)
            for _ in range(500):
                i = torch.randint(0, len(a0), (256,), device=dev)
                with torch.no_grad():
                    s0, s1t = model.encode(a0[i]), model.encode(a1[i])
                s1h = model.fwd(s0, model.idm(a0[i], a1[i]))
                l = mse(model.decode(s1h), a1[i]) + mse(s1h, s1t)
                opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad():
            pred = model.decode(model.fwd(model.encode(o0b), model.idm(o0b, o1b)))
            res["fewshot_fwd_rel_err"][k] = (mse(pred, o1b) / mse(o0b, o1b)).item()

os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)
with open(os.path.join(ROOT, args.out, f"{A}_{args.fold}_{kind}.json"), "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps(res))
