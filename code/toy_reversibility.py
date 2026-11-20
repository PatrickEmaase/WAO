"""Controlled study with known, state-dependent dissipative drift and ground-truth single-step reversibility.

World (first order, as in Definition 1):  p_{t+1} = p_t + dt * (f(p_t) + u_t),   p in R^2, u in [-1, 1]^2
    f(p) = g - k * w(p) * (p - c),   g = (0, -0.5),  w(p) = sigmoid((r0 - |p - c|) / sigma)
i.e. a constant gravity-like drift everywhere plus a dissipative well around c in which the scene contracts
("settles") on its own. The agent may exert up to U_MAX = 2 per axis when undoing, so outside the well every
step can be undone in one step, while inside the well the contraction can exceed what the agent can reverse.

Ground truth for a transition (p_t, u_t, p_{t+1}):  the unique one-step undo action is
    u* = (p_t - p_{t+1}) / dt - f(p_{t+1}),   and the step is single-step reversible iff max|u*| <= U_MAX.

Models see only o = lift(p) (a fixed random 2-layer tanh lift to 32-D), never p or u. Idle steps (u = 0) occur with
probability 0.15; as on the robot, the anchor only sees idle steps outside the well (idle and strong drift rarely
coincide). Variants mirror rave.py: rave (RAVI), rave_noanchor, rave_sr (swapped pair under +G0), rave_norev,
rave_nodrift (G0 = 0), lam (MLP + separate backward model), signrev (MLP, z_bar = -z).

Usage: python toy_reversibility.py [--seeds 0 1 2] [--steps 4000] [--out ../results/toy_reversibility.json]
"""
import argparse, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))          # revision 2: 10 seeds
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--models", nargs="+", default=["rave", "rave_noanchor", "rave_sr", "rave_norev", "rave_nodrift",
                                                "lam", "signrev", "rave_sr_noanchor", "rave_nodrift_noanchor",
                                                "rave_2f"])
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                                             "toy_reversibility.json"))
p.add_argument("--run_dir", default=None)     # one JSON per (model, seed); existing runs are skipped, then aggregated
p.add_argument("--aggregate_only", action="store_true")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")    # revision 2: GPU
args = p.parse_args()
torch.set_num_threads(4)
DEV = torch.device(args.device)

DT, U_MAX, G = 0.1, 2.0, torch.tensor([0.0, -0.5])
C, R0, SIG, K = torch.tensor([0.0, -1.0]), 1.0, 0.1, 3.0
OBS, D, M = 32, 8, 2

torch.manual_seed(12345)                                  # the lift is part of the world: fixed across seeds
lift = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, OBS), nn.Tanh()).requires_grad_(False)


def drift(p):
    c = C.to(p.device)
    w = torch.sigmoid((R0 - (p - c).norm(dim=-1, keepdim=True)) / SIG)
    return G.to(p.device) - K * w * (p - c)


def in_well(p):
    return (p - C.to(p.device)).norm(dim=-1) < R0


def sample(n, gen):
    p = torch.rand(n, 2, generator=gen) * 4 - 2
    u = torch.rand(n, 2, generator=gen) * 2 - 1
    idle = torch.rand(n, generator=gen) < 0.15
    u[idle] = 0
    p1 = p + DT * (drift(p) + u)
    return p, u, p1, idle


def undo_truth(p, p1):
    return (p - p1) / DT - drift(p1)


mlp = lambda i, o: nn.Sequential(nn.Linear(i, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, o))


def batch(n, gen, two_frame=False):
    """Inputs and targets of one batch. Single-frame models see o_t; the two-frame (velocity) variant sees
    [o_t, o_t - o_{t-1}] and, for the swapped pair, the time-reversed velocities [o_{t+1}, o_{t+1} - o_{t+2}]."""
    to = lambda d: {k: v.to(DEV) for k, v in d.items()}                    # sampling on the CPU, compute on DEV
    if not two_frame:                                                       # identical random draws as revision 1
        p, u, p1, idle = sample(n, gen); o0, o1 = lift(p), lift(p1)
        return to(dict(p=p, u=u, p1=p1, idle=idle, o0=o0, o1=o1, X0=o0, X1=o1, X1r=o1, X0r=o0))
    pm = torch.rand(n, 2, generator=gen) * 4 - 2; um = torch.rand(n, 2, generator=gen) * 2 - 1
    p = pm + DT * (drift(pm) + um)                                          # a short trajectory p_{t-1} -> p_{t+2}
    u = torch.rand(n, 2, generator=gen) * 2 - 1
    idle = torch.rand(n, generator=gen) < 0.15
    u[idle] = 0
    p1 = p + DT * (drift(p) + u)
    u2 = torch.rand(n, 2, generator=gen) * 2 - 1; p2 = p1 + DT * (drift(p1) + u2)
    om, o0, o1, o2 = lift(pm), lift(p), lift(p1), lift(p2)
    return to(dict(p=p, u=u, p1=p1, idle=idle, o0=o0, o1=o1,
                   X0=torch.cat([o0, o0 - om], -1), X1=torch.cat([o1, o1 - o0], -1),
                   X1r=torch.cat([o1, o1 - o2], -1), X0r=torch.cat([o0, o0 - o1], -1)))


def parse_kind(kind):
    """Model variant -> (core variant, anchored?, two-frame?). Revision 2 adds unanchored SR and G0 = 0 variants
    and a two-frame (velocity) RAVI."""
    two = kind == "rave_2f"
    noanchor = kind.endswith("_noanchor")
    core = "rave" if two else (kind[: -len("_noanchor")] if noanchor else kind)
    return core, not noanchor, two


class Base(nn.Module):
    def __init__(self, inp=OBS):
        super().__init__()
        self.enc, self.dec, self.idm_net = mlp(inp, D), mlp(D, OBS), mlp(3 * inp, M)

    def idm(self, a, b): return self.idm_net(torch.cat([a, b, b - a], -1))


class LAM(Base):
    def __init__(self):
        super().__init__(); self.fdm, self.bdm = mlp(D + M, D), mlp(D + M, D)
    def encode(self, o): return self.enc(o)
    def decode(self, s): return self.dec(s)
    def fwd(self, s, z): return s + self.fdm(torch.cat([s, z], -1))
    def bwd(self, s, z): return s + self.bdm(torch.cat([s, z], -1))


class RAVE(Base):
    def __init__(self, use_drift=True, inp=OBS):
        super().__init__(inp); self.use_drift = use_drift
        self.W = nn.Parameter(0.01 * torch.randn(M, D, D)); self.b = nn.Parameter(0.01 * torch.randn(M, D))
        self.V = nn.Parameter(torch.zeros(M, D, D))
        self.S0 = nn.Parameter(0.01 * torch.randn(D, D)); self.L0 = nn.Parameter(0.01 * torch.randn(D, D))
        self.b0 = nn.Parameter(torch.zeros(D))

    def encode(self, o): s = self.enc(o); return torch.cat([s, torch.ones_like(s[:, :1])], -1)
    def decode(self, s): return self.dec(s[:, :-1])

    def gens(self):
        Gi = torch.zeros(M, D + 1, D + 1, device=self.W.device)
        Gi[:, :D, :D] = (self.W - self.W.transpose(1, 2)) + 0.5 * (self.V + self.V.transpose(1, 2)); Gi[:, :D, D] = self.b
        G0 = torch.zeros(D + 1, D + 1, device=self.W.device)
        G0[:D, :D] = (self.S0 - self.S0.T) - self.L0 @ self.L0.T; G0[:D, D] = self.b0
        return (G0 if self.use_drift else torch.zeros_like(G0)), Gi

    def gen(self, z, drift=1.0):
        G0, Gi = self.gens(); return drift * G0 + torch.einsum("bm,mij->bij", z, Gi)

    @staticmethod
    def step(A, s): return (torch.matrix_exp(A) @ s.unsqueeze(-1)).squeeze(-1)
    def fwd(self, s, z): return self.step(self.gen(z), s)

    def drift_action(self, S):
        G0, Gi = self.gens()
        X = torch.einsum("mij,nj->nim", Gi, S).reshape(-1, M); y = (S @ G0.T).reshape(-1, 1)
        if y.norm() == 0:
            return torch.zeros(M, device=S.device), float("nan")
        c = torch.linalg.lstsq(X, y).solution.squeeze(-1)
        return c, ((y.squeeze(-1) - X @ c).norm() / y.norm()).item()


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train(kind, seed):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    core, anchored, two = parse_kind(kind)
    model = LAM() if core in ("lam", "signrev") else RAVE(use_drift=(core != "rave_nodrift"), inp=2 * OBS if two else OBS)
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); mse = F.mse_loss
    for _ in range(args.steps):
        B = batch(512, gen, two); pp, idle, o0, o1 = B["p"], B["idle"], B["o0"], B["o1"]
        anchor = idle & ~in_well(pp)                                        # idle steps seen by the anchor
        s0, s1 = model.encode(B["X0"]), model.encode(B["X1"])
        z = model.idm(B["X0"], B["X1"]); zn = z + 0.05 * torch.randn_like(z); s1h = model.fwd(s0, zn)
        L = (mse(model.decode(s0), o0) + mse(model.decode(s1), o1) + mse(model.decode(s1h), o1) + mse(s1h, s1)
             + 1e-3 * z.pow(2).mean())
        zb = model.idm(B["X1r"], B["X0r"]); s1r = s1 if not two else model.encode(B["X1r"])
        if core == "lam":
            L = L + mse(model.decode(model.bwd(s1r, zb)), o0)
        elif core == "signrev":
            L = L + mse(model.decode(model.fwd(s1r, zb)), o0) + mse(zb, -z)
        else:
            L = L + mse(model.decode(model.step(-model.gen(zn), s1)), o0)
            if core == "rave_sr":
                L = L + mse(model.decode(model.step(model.gen(zb, 1.0), s1r)), o0) + mse(zb, -z)
            elif core != "rave_norev":
                L = L + mse(model.decode(model.step(model.gen(zb, -1.0), s1r)), o0) + mse(zb, -z)
            if anchored and anchor.any():
                L = L + z[anchor].pow(2).mean()
            L = L + 1e-3 * model.V.pow(2).sum()
        opt.zero_grad(); L.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model.eval()


def roll(model, s, zs):
    for z in zs:
        s = model.fwd(s, z)
    return s


def evaluate(model, kind, seed, n=6000):
    gen = torch.Generator().manual_seed(10_000 + seed)
    koop = isinstance(model, RAVE); r = {}
    core, _, two = parse_kind(kind)
    B = batch(n, gen, two); pp, u, p1, idle, o0, o1 = B["p"], B["u"], B["p1"], B["idle"], B["o0"], B["o1"]
    X0, X1, X1r, X0r = B["X0"], B["X1"], B["X1r"], B["X0r"]
    ustar = undo_truth(pp, p1); irrev = (ustar.abs().amax(1) > U_MAX)
    r["frac_irreversible"] = irrev.float().mean().item()
    with torch.no_grad():
        s0, s1 = model.encode(X0), model.encode(X1); z = model.idm(X0, X1)
        rel = lambda pred, tgt, other: (F.mse_loss(pred, tgt) / F.mse_loss(other, tgt)).item()
        r["forward_rel_err"] = rel(model.decode(model.fwd(s0, z)), o1, o0)
        if core == "lam":
            ob = model.decode(model.bwd(model.encode(X1r), model.idm(X1r, X0r)))
        elif core == "signrev":
            ob = model.decode(model.fwd(s1, -z))
        else:
            ob = model.decode(model.step(-model.gen(z), s1))
        r["backward_rel_err"] = rel(ob, o0, o1)
        # gauge-fixing linear probe z -> u, fitted on labeled transitions from separate data (as in the original toy)
        Bq = batch(8000, torch.Generator().manual_seed(20_000 + seed), two); uq = Bq["u"]
        Zq = model.idm(Bq["X0"], Bq["X1"]); Xq = torch.cat([Zq, torch.ones(len(Zq), 1, device=DEV)], -1)
        Wp = torch.linalg.lstsq(Xq, uq).solution
        probe = lambda zz: torch.cat([zz, torch.ones(len(zz), 1, device=DEV)], -1) @ Wp
        uh = probe(z)
        r["probe_R2"] = (1 - ((uh - u) ** 2).sum() / ((u - u.mean(0)) ** 2).sum()).item()
        # drift absorption: idle steps inside the well (u = 0, strong passive motion)
        iw = idle & in_well(pp); io = idle & ~in_well(pp)
        rms_u = u[~idle].norm(dim=1).mean()
        r["absorbed_in_well"] = (uh[iw].norm(dim=1).mean() / rms_u).item()
        r["absorbed_outside"] = (uh[io].norm(dim=1).mean() / rms_u).item()
        # reversal consistency of the IDM
        r["cos_idm_swap_negz"] = F.cosine_similarity(model.idm(X1r, X0r), -z).mean().item()
        # reversibility scores against the ground-truth label
        sc = {"feat_change": (o1 - o0).norm(dim=1), "znorm": z.norm(dim=1), "true_drift": drift(p1).norm(dim=1)}
        if koop:
            A_ = model.gen(z); sv = torch.linalg.svdvals(torch.matrix_exp(A_)); sc["kappa"] = sv[:, 0] / sv[:, -1]
            sv = torch.linalg.svdvals(torch.matrix_exp(A_[:, :D, :D])); sc["kappa_lin"] = sv[:, 0] / sv[:, -1]
            if model.use_drift:
                G0, Gi = model.gens(); c, eps = model.drift_action(s0); r["drift_residual"] = eps
                x = s1[:, :D]; sc["rho"] = (x @ model.L0).pow(2).sum(1) / x.pow(2).sum(1)
                Gc = G0 - torch.einsum("m,mij->ij", c, Gi)
                sc["eps_t"] = (s1 @ Gc.T).norm(dim=1) / (s1 @ G0.T).norm(dim=1)
                vals = []
                for k in range(5):                                          # random drift of matched structure and norm
                    gk = torch.Generator().manual_seed(30_000 + 10 * seed + k)
                    Sr, Lr, br = torch.randn(D, D, generator=gk), torch.randn(D, D, generator=gk), torch.randn(D, generator=gk)
                    Gr = torch.zeros(D + 1, D + 1); Gr[:D, :D] = Sr - Sr.T - Lr @ Lr.T; Gr[:D, D] = br; Gr = Gr.to(DEV)
                    Gr = Gr * (G0.norm() / Gr.norm())
                    X = torch.einsum("mij,nj->nim", Gi, s0).reshape(-1, M); y = (s0 @ Gr.T).reshape(-1, 1)
                    cr = torch.linalg.lstsq(X, y).solution; vals.append(((y - X @ cr).norm() / y.norm()).item())
                r["drift_residual_randomG0"] = float(np.mean(vals))
    if not koop:                                                            # MLP: condition number of d s_{t+1} / d s_t
        J = torch.func.vmap(torch.func.jacrev(lambda x, zz: model.fwd(x[None], zz[None])[0]))(s0.detach(), z.detach())
        sv = torch.linalg.svdvals(J); sc["kappa"] = (sv[:, 0] / sv[:, -1]).detach()
    # motion-matched negatives: reversible steps with the same true action magnitude
    pos_i, neg_i = torch.where(irrev)[0].cpu().numpy(), torch.where(~irrev)[0].cpu().numpy()
    un, dn = u.norm(dim=1).cpu().numpy(), (p1 - pp).norm(dim=1).cpu().numpy()
    mm = np.concatenate([neg_i[np.argsort(np.abs(un[neg_i] - un[i]))[:3]] for i in pos_i])   # same agent motion
    dm = np.concatenate([neg_i[np.argsort(np.abs(dn[neg_i] - dn[i]))[:3]] for i in pos_i])   # same displacement
    for k, v in sc.items():
        v = v.detach().cpu().numpy()
        r[f"auroc_{k}"] = auroc(v[pos_i], v[neg_i]); r[f"auroc_mm_{k}"] = auroc(v[pos_i], v[mm])
        r[f"auroc_dm_{k}"] = auroc(v[pos_i], v[dm])
    # undo: model's closed-form reverse action and state-specific planned undo, executed in the true simulator
    sub = torch.arange(0, n, 6, device=DEV)                                             # 1,000 transitions
    o0s, ps, p1s, us = o0[sub], pp[sub], p1[sub], u[sub]; irs = irrev[sub]
    with torch.no_grad():
        s1s = model.encode(X1[sub]); zs = model.idm(X0[sub], X1[sub])
        if core == "lam":
            z_init = model.idm(X1r[sub], X0r[sub])
        elif core in ("signrev", "rave_sr"):
            z_init = -zs
        else:
            z_init = -zs - 2 * model.drift_action(model.encode(X0))[0]
    zp = z_init.clone().requires_grad_(True); opt = torch.optim.Adam([zp], lr=0.05)
    for _ in range(300):
        loss = (model.decode(model.fwd(s1s, zp)) - o0s).pow(2).sum(1).mean()
        (gz,) = torch.autograd.grad(loss, [zp]); zp.grad = gz; opt.step()
    step_len = (p1s - ps).norm(dim=1)

    def ret(ua):                                                            # simulator return error / step length
        ua = ua.clamp(-U_MAX, U_MAX)
        return ((p1s + DT * (drift(p1s) + ua) - ps).norm(dim=1) / step_len)
    with torch.no_grad():
        errs = {"model_closed": ret(probe(z_init)), "model_planned": ret(probe(zp.detach())),
                "naive_neg_u": ret(-us), "best_possible": ret(undo_truth(ps, p1s))}
    for k, e in errs.items():
        r[f"return_{k}_reversible"] = e[~irs].median().item()
        r[f"return_{k}_irreversible"] = e[irs].median().item()
    return r


run_dir = args.run_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "toy_runs")
os.makedirs(run_dir, exist_ok=True)
if not args.aggregate_only:
    for kind in args.models:
        for seed in args.seeds:
            f = os.path.join(run_dir, f"{kind}_s{seed}.json")
            if os.path.exists(f):
                continue
            r = evaluate(train(kind, seed), kind, seed); r.update(model=kind, seed=seed, steps=args.steps)
            json.dump(r, open(f, "w"), indent=1)
            print(kind, seed, {k: round(v, 3) for k, v in r.items() if k in ("forward_rel_err", "auroc_kappa",
                  "absorbed_in_well", "return_model_planned_reversible", "drift_residual")}, flush=True)
runs = {}
for f in sorted(os.listdir(run_dir)):
    r = json.load(open(os.path.join(run_dir, f))); runs.setdefault(r["model"], []).append(r)
out = {}
for kind, rs in runs.items():
    keys = sorted(k for k in set().union(*rs) if k not in ("model", "seed", "steps"))
    from scipy import stats as _st                                          # [mean, std, n, 95% CI half-width]
    def summ(v):
        v = np.array(v, float); n_ = len(v)
        ci = float(_st.t.ppf(0.975, n_ - 1) * v.std(ddof=1) / np.sqrt(n_)) if n_ > 1 else float("nan")
        return [float(v.mean()), float(v.std()), n_, ci]
    out[kind] = {k: summ([r[k] for r in rs if k in r]) for k in keys}
json.dump({"config": vars(args), "seeds": {k: sorted(r["seed"] for r in v) for k, v in runs.items()}, "results": out},
          open(args.out, "w"), indent=1)
print("wrote", os.path.abspath(args.out))
