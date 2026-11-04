"""Toy check of a Bidirectional Koopman Latent Action Model (BiK-LAM).

World: a 2-D point with constant drift g (gravity / wind / conveyor) and an action u:
    p_{t+1} = p_t + dt * (g + u)
The model only sees a fixed random nonlinear lift o = lift(p), never p or u.

Model:  s = [enc(o), 1]                       (constant homogeneous coordinate -> translations are linear)
        z = IDM(o_t, o_{t+1})                 (latent action)
        forward   s_{t+1} = exp(G0 + sum_i z_i G_i) s_t
        backward  s_t     = exp(-(G0 + sum_i z_i G_i)) s_{t+1}   (exact inverse, no extra parameters)

Trained on FORWARD transitions only. Backward prediction and the "undo" action are derived, then
checked against held-out data and against the true simulator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DT, G_DRIFT = 0.1, torch.tensor([0.0, -0.5])
OBS, D, M = 32, 8, 2          # observation dim, latent dim (excl. homogeneous coord), latent-action dim

torch.manual_seed(0)
lift = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, OBS), nn.Tanh()).requires_grad_(False)


def sample(n):
    p = torch.rand(n, 2) * 4 - 2
    u = torch.rand(n, 2) * 2 - 1
    return p, u, p + DT * (G_DRIFT + u)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, o))


class BiKLAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec, self.idm = mlp(OBS, D), mlp(D, OBS), mlp(2 * OBS, M)
        self.G0 = nn.Parameter(torch.zeros(D + 1, D + 1))                # drift generator (passive dynamics)
        self.G = nn.Parameter(0.01 * torch.randn(M, D + 1, D + 1))      # one generator per latent-action dim
        mask = torch.ones(D + 1, D + 1)
        mask[-1] = 0                                                     # keeps the homogeneous coord at 1
        self.register_buffer("mask", mask)

    def encode(self, o):
        s = self.enc(o)
        return torch.cat([s, torch.ones_like(s[:, :1])], -1)

    def decode(self, s):
        return self.dec(s[:, :-1])

    def gen(self, z):                                                    # z: (B, M) -> (B, D+1, D+1)
        return (self.G0 + torch.einsum("bm,mij->bij", z, self.G)) * self.mask

    @staticmethod
    def step(A, s):                                                      # s <- exp(A) s
        return (torch.matrix_exp(A) @ s.unsqueeze(-1)).squeeze(-1)

    def undo_code(self, s1, s0, z_init, iters=300):
        """Latent action that takes s_{t+1} back to s_t under the FORWARD model: argmin_z' ||K(z') s1 - s0||."""
        z = z_init.clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=0.05)
        for _ in range(iters):
            loss = F.mse_loss(self.step(self.gen(z), s1), s0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return z.detach()


def train(use_rev, steps=6000, bs=512):
    torch.manual_seed(1)
    model = BiKLAM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        p, u, p1 = sample(bs)
        o0, o1 = lift(p), lift(p1)
        s0, s1 = model.encode(o0), model.encode(o1)
        z = model.idm(torch.cat([o0, o1], -1))
        s1_hat = model.step(model.gen(z), s0)
        loss = (F.mse_loss(model.decode(s0), o0) + F.mse_loss(model.decode(s1), o1)
                + F.mse_loss(s1_hat, s1) + F.mse_loss(model.decode(s1_hat), o1))
        if use_rev:                                    # IDM on the time-reversed pair should give -z
            z_rev = model.idm(torch.cat([o1, o0], -1))
            loss = loss + F.mse_loss(z_rev, -z)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def evaluate(model, n=8000):
    torch.manual_seed(123)
    p, u, p1 = sample(n)
    o0, o1 = lift(p), lift(p1)
    with torch.no_grad():
        s0, s1 = model.encode(o0), model.encode(o1)
        z = model.idm(torch.cat([o0, o1], -1))
        A = model.gen(z)
        z_rev = model.idm(torch.cat([o1, o0], -1))
        z_noop = model.idm(torch.cat([o0, lift(p + DT * G_DRIFT)], -1))   # pure-drift step (u = 0)
        rel = lambda pred, tgt, other: (F.mse_loss(pred, tgt) / F.mse_loss(other, tgt)).item()
        r = {
            "forward  rel. error   exp(A) s_t": rel(model.decode(model.step(A, s0)), o1, o0),
            "backward rel. error   exp(-A) s_t+1  (never trained)": rel(model.decode(model.step(-A, s1)), o0, o1),
            "cos( IDM(o_t+1, o_t), -z )": F.cosine_similarity(z_rev, -z).mean().item(),
            "|z_noop| / rms(z)   (drift encoded as action?)": (z_noop.norm(dim=-1).mean() / z.norm(dim=-1).pow(2).mean().sqrt()).item(),
        }
        # Linear probe latent action -> true action (needs labels: this is what fixes the drift/action gauge).
        X = torch.cat([z, torch.ones(n, 1)], -1)
        W = torch.linalg.lstsq(X, u).solution
        r["R^2 of linear probe z -> u"] = (1 - ((X @ W - u) ** 2).sum() / ((u - u.mean(0)) ** 2).sum()).item()

    # Undo action under forward-time physics, found by planning in the model, executed in the TRUE simulator.
    z_undo = model.undo_code(s1, s0, -z)
    with torch.no_grad():
        u_undo = torch.cat([z_undo, torch.ones(n, 1)], -1) @ W
        u_star = -u - 2 * G_DRIFT                                         # analytic undo action
        step_len = (p1 - p).norm(dim=-1).mean()
        back = lambda ua: ((p1 + DT * (G_DRIFT + ua) - p).norm(dim=-1).mean() / step_len).item()
        r["sim return error / step:  model undo action"] = back(u_undo)
        r["sim return error / step:  naive  u' = -u   "] = back(-u)
        r["mean |u_undo - (-u - 2g)|  (naive -u gives 0.5)"] = (u_undo - u_star).abs().mean().item()
    return r


if __name__ == "__main__":
    for use_rev in (False, True):
        print(f"\n=== trained on forward data only, reversal-consistency loss = {use_rev} ===")
        for k, v in evaluate(train(use_rev)).items():
            print(f"  {k:<56s} {v:10.5f}")
