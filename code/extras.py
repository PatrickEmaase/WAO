"""Revision-1 diagnostics for rave.py.

Called once at the end of rave.py as run_extras(globals()), after every original metric has been computed, with the
model restored to its non-adapted weights. Randomness comes from a separate generator seeded by (seed, fold), so the
original metrics of earlier runs are reproduced exactly. Every quantity below is described in the revised paper
(Section VI-D) and in REVISION_CHECKLIST.md.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F


def _cos_along(path, za, zb, n):
    al = [(k, l) for k, l in path if k < n and l < n]
    ka, la = np.array([a for a, _ in al]), np.array([b for _, b in al])
    return float(np.mean(np.sum(za[ka] * zb[la], 1)
                         / (np.linalg.norm(za[ka], axis=1) * np.linalg.norm(zb[la], axis=1) + 1e-8)))


def _logreg_auroc(Xtr, ytr, Xte, yte, auroc, l2=1e-2, iters=200):
    """L2-regularised logistic regression (LBFGS, float64); AUROC on the test set."""
    if ytr.sum() == 0 or ytr.sum() == len(ytr) or yte.sum() == 0 or yte.sum() == len(yte):
        return None
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_, Xte_ = torch.tensor((Xtr - mu) / sd), torch.tensor((Xte - mu) / sd)
    ytr_ = torch.tensor(ytr, dtype=torch.float64)
    w = torch.zeros(Xtr_.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
    pw = (len(ytr) - ytr.sum()) / ytr.sum()                                    # balance the rare positives

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xtr_ @ w + b, ytr_, pos_weight=torch.tensor(pw)) + l2 * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    s = (Xte_ @ w + b).detach().numpy()
    return float(auroc(s[yte == 1], s[yte == 0]))


def run_extras(g):
    model, kind, args, res = g["model"], g["kind"], g["args"], g["res"]
    DA, DB, teA, teB, trA, trB = g["DA"], g["DB"], g["teA"], g["teB"], g["trA"], g["trB"]
    T, auroc, release_carry, cond_scores = g["T"], g["auroc"], g["release_carry"], g["cond_scores"]
    ep_actions, reverse_actions, rollout, dtw_path = g["ep_actions"], g["reverse_actions"], g["rollout"], g["dtw_path"]
    tgt, Ztr, d, mse = g["tgt"], g["Ztr"], g["d"], g["mse"]
    koop = kind in g["KOOPMAN"]
    has_drift = koop and getattr(model, "use_drift", False)
    rng = np.random.RandomState(1000 + 97 * args.seed + args.fold)
    torch.manual_seed(1000 + 97 * args.seed + args.fold)
    ex = {}

    # ------------------------------------------------ 1. null rollouts and the permuted alignment reference
    with torch.no_grad():
        mu_z, cov_z = Ztr.mean(0), np.cov(Ztr.T) + 1e-6 * np.eye(Ztr.shape[1])
        if koop:
            c, _ = model.drift_action()
        D = {k: [] for k in ("null_zero", "null_random", "null_forward", "negz", "undo")}
        cos_perm = []
        rel_null, end_dec, null_dec = [], [], []
        for j, eb in enumerate(teB):
            ea = teA[j % len(teA)]
            oa, za = ep_actions(DA, ea); ob_, zb = ep_actions(DB, eb)
            d0 = np.linalg.norm(DB["o"][eb][0] - tgt)
            end = lambda zs: np.linalg.norm(rollout(ob_[0], zs).cpu().numpy()[-1] - tgt) / d0
            D["null_zero"].append(end(torch.zeros_like(za)))
            zr_rand = torch.tensor(rng.multivariate_normal(mu_z, cov_z, len(za)), dtype=za.dtype, device=za.device)
            D["null_random"].append(end(zr_rand))
            D["null_forward"].append(end(za))
            zr_m = zb if kind == "oracle" else reverse_actions(oa, za)                 # revision 2
            m_end, z_end = end(zr_m), end(torch.zeros_like(za))
            rel_null.append(m_end / z_end)
            d0_dec = np.linalg.norm(rollout(ob_[0], za[:0]).cpu().numpy()[-1] - tgt)  # decoded start state
            end_dec.append(np.linalg.norm(rollout(ob_[0], zr_m).cpu().numpy()[-1] - tgt) / d0_dec)
            null_dec.append(np.linalg.norm(rollout(ob_[0], torch.zeros_like(za)).cpu().numpy()[-1] - tgt) / d0_dec)
            if koop:
                D["negz"].append(end((-za).flip(0)))
                D["undo"].append(end((-za - 2 * c).flip(0)))
            zr = zb if kind == "oracle" else reverse_actions(oa, za)
            path = dtw_path(DA["o"][ea][::-1], DB["o"][eb])
            zb_np = zb.cpu().numpy()
            cos_perm.append(_cos_along(path, zr.cpu().numpy(), zb_np[rng.permutation(len(zb_np))], len(zr)))
        for k, v in D.items():
            if v:
                ex[f"end_dist_ratio_{k}"] = float(np.mean(v))
        ex["action_cos_permuted"] = float(np.mean(cos_perm))
        ex["end_dist_rel_null"] = float(np.mean(rel_null))                 # own end distance / own zero-action null
        ex["end_dist_ratio_decstart"] = float(np.mean(end_dec))            # measured from the decoded start state
        ex["end_dist_ratio_null_zero_decstart"] = float(np.mean(null_dec))

    # ------------------------------------------------ 2. reversibility detectors, ordinary and motion-matched negatives
    scores = {k: ([], []) for k in ("kappa", "kappa_lin", "rho", "eps_t", "znorm", "feat_change", "arm_speed",
                                    "gripper_open", "inv_err", "inv_err_abs")}
    spd_pos, spd_neg, dfe_pos, dfe_neg = [], [], [], []
    trans = {"release": [], "carry": []}                                            # (o_t, o_{t+1}) for the undo test
    with torch.no_grad():
        if has_drift:
            G0, Gi = model.gens(); c, _ = model.drift_action()
            Gc = G0 - torch.einsum("m,mij->ij", c, Gi)
        for e in teA:
            o = T(DA["o"][e]); n = len(o) - 1
            rel, carry = release_carry(DA, e, n)
            z = model.idm(o[:-1], o[1:])
            sc = {"kappa": cond_scores(o),
                  "znorm": z.norm(dim=1).cpu().numpy(),
                  "feat_change": (o[1:] - o[:-1]).norm(dim=1).cpu().numpy(),
                  "arm_speed": DA["speed"][e][:n]}
            gq = DA["qpos"][e][:, [6, 13]]
            lo, hi = np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, len(gq) - 1)
            sc["gripper_open"] = (gq[hi] - gq[lo]).max(1)
            s1e = model.encode(o[1:])                                                   # revision 2: inverse-step error
            if kind == "lam":
                ob_inv = model.decode(model.bwd(s1e, model.idm(o[1:], o[:-1])))
            elif kind == "signrev":
                ob_inv = model.decode(model.fwd(s1e, -z))
            else:
                ob_inv = model.decode(model.step(-model.gen(z), s1e))
            ie = (ob_inv - o[:-1]).norm(dim=1)
            sc["inv_err_abs"] = ie.cpu().numpy()
            sc["inv_err"] = (ie / (o[1:] - o[:-1]).norm(dim=1)).cpu().numpy()
            if koop:
                A_ = model.gen(z); M = A_[:, :d, :d]
                sv = torch.linalg.svdvals(torch.matrix_exp(M)); sc["kappa_lin"] = (sv[:, 0] / sv[:, -1]).cpu().numpy()
            if has_drift:
                s = model.encode(o[:-1]); x = s[:, :d]
                sc["rho"] = ((x @ model.L0).pow(2).sum(1) / x.pow(2).sum(1)).cpu().numpy()
                sc["eps_t"] = ((s @ Gc.T).norm(dim=1) / (s @ G0.T).norm(dim=1)).cpu().numpy()
            for k, v in sc.items():
                scores[k][0].extend(v[rel]); scores[k][1].extend(v[carry])
            spd_pos.extend(DA["speed"][e][:n][rel]); spd_neg.extend(DA["speed"][e][:n][carry])
            dfe_pos.extend(sc["feat_change"][rel]); dfe_neg.extend(sc["feat_change"][carry])
            for t in np.where(rel)[0]:
                trans["release"].append((DA["o"][e][t], DA["o"][e][t + 1]))
            for t in np.where(carry)[0]:
                trans["carry"].append((DA["o"][e][t], DA["o"][e][t + 1]))
    spd_pos, spd_neg = np.array(spd_pos), np.array(spd_neg)
    mm = np.concatenate([np.argsort(np.abs(spd_neg - sp))[:3] for sp in spd_pos]) if len(spd_pos) and len(spd_neg) else []
    dfe_pos, dfe_neg = np.array(dfe_pos), np.array(dfe_neg)
    dm = np.concatenate([np.argsort(np.abs(dfe_neg - dp))[:3] for dp in dfe_pos]) if len(dfe_pos) and len(dfe_neg) else []
    ex["n_motion_matched"] = int(len(mm))
    for k, (p_, n_) in scores.items():
        if p_ and n_:
            p_, n_ = np.array(p_, float), np.array(n_, float)
            ex[f"auroc_{k}"] = float(auroc(p_, n_))
            ex[f"auroc_mm_{k}"] = float(auroc(p_, n_[mm]))                 # same arm speed (agent motion)
            ex[f"auroc_dm_{k}"] = float(auroc(p_, n_[dm]))                 # same feature change (total motion)
            if k == "eps_t":
                ex["eps_t_release_mean"], ex["eps_t_carry_mean"] = float(p_.mean()), float(n_.mean())

    # ------------------------------------------------ 3. drift-residual reference: random G0 of matched structure and norm
    if has_drift:
        with torch.no_grad():
            G0, Gi = model.gens(); S = model.S_ref; vals = []
            X = torch.einsum("mij,nj->nim", Gi, S).reshape(-1, Gi.shape[0])
            for _ in range(5):
                Sr, Lr, br = (torch.tensor(rng.randn(*sh), dtype=G0.dtype, device=G0.device) for sh in ((d, d), (d, d), (d,)))
                Gr = torch.zeros_like(G0); Gr[:d, :d] = Sr - Sr.T - Lr @ Lr.T; Gr[:d, d] = br
                Gr = Gr * (G0.norm() / Gr.norm())
                y = (S @ Gr.T).reshape(-1, 1); cr = torch.linalg.lstsq(X, y).solution
                vals.append(((y - X @ cr).norm() / y.norm()).item())
            ex["drift_residual_randomG0"] = float(np.mean(vals))

    # ------------------------------------------------ 4. state-specific and H-step undo (decoded feature space)
    for name, tr in trans.items():
        if not tr:
            continue
        o0 = T(np.stack([a for a, _ in tr])); o1 = T(np.stack([b for _, b in tr]))
        with torch.no_grad():
            s1 = model.encode(o1); z = model.idm(o0, o1); base = (o1 - o0).norm(dim=1)
            if kind == "lam":
                z_init = model.idm(o1, o0)
            elif kind in ("signrev", "rave_sr"):
                z_init = -z
            else:
                z_init = -z - 2 * model.drift_action()[0]
            rel_err = lambda zs: (model.decode(_roll(model, s1, zs)) - o0).norm(dim=1) / base
            abs_err = lambda zs: (model.decode(_roll(model, s1, zs)) - o0).norm(dim=1)
            ex[f"undo_closed_relerr_{name}"] = float(rel_err([z_init]).median())
            ex[f"undo_closed_abserr_{name}"] = float(abs_err([z_init]).median())
            ex[f"step_len_{name}"] = float(base.median())
        for H, key in ((1, "undo1"), (args.undo_H, "undoH")):
            zs = [z_init.clone()] + [torch.zeros_like(z_init) for _ in range(H - 1)]
            zs = [zz.requires_grad_(True) for zz in zs]
            opt = torch.optim.Adam(zs, lr=0.05)
            for _ in range(args.undo_iters):
                with torch.enable_grad():
                    loss = (model.decode(_roll(model, s1, zs)) - o0).pow(2).sum(1).mean()
                    grads = torch.autograd.grad(loss, zs)
                for zz, gg in zip(zs, grads):
                    zz.grad = gg
                opt.step()
            with torch.no_grad():
                r = rel_err([zz.detach() for zz in zs])
                ra = abs_err([zz.detach() for zz in zs])
            ex[f"{key}_abserr_{name}"] = float(ra.median())
            ex[f"{key}_relerr_{name}"] = float(r.median())
            ex[f"{key}_success_{name}"] = float((r < args.undo_tau).float().mean())

    # ------------------------------------------------ 5. size of the generator, strip condition of Propositions 4 and 6
    if koop:
        with torch.no_grad():
            o0, o1, _ = g["pairs"](DA, teA); z = model.idm(o0, o1)
            c = model.drift_action()[0]
            nA, strip = [], []
            for A_ in (model.gen(z), model.gen(-z - 2 * c)):
                nA.append(torch.linalg.matrix_norm(A_, ord=2))
                strip.append(torch.linalg.eigvals(A_).imag.abs().amax(-1) < np.pi)
            nA = torch.cat(nA)
            ex["A_norm_median"], ex["A_norm_max"] = float(nA.median()), float(nA.max())
            ex["strip_frac"] = float(torch.cat(strip).float().mean())

    # ------------------------------------------------ 6. drift absorption: how linearly decodable is a release from z_t?
    def zfeat(D, eps):
        X, y, Xd, Xm = [], [], [], []
        with torch.no_grad():
            for e in eps:
                o = T(D["o"][e]); n = len(o) - 1; rel, carry = release_carry(D, e, n); keep = rel | carry
                X.append(model.idm(o[:-1], o[1:]).cpu().numpy()[keep]); y.append(rel[keep].astype(float))
                Xd.append((o[1:] - o[:-1]).cpu().numpy()[keep])
                gq = D["qpos"][e][:, [6, 13]]                                  # arm and gripper motion of each step
                lo, hi = np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, len(gq) - 1)
                Xm.append(np.c_[D["dq"][e][:n], D["speed"][e][:n], gq[hi] - gq[lo]][keep])
        cat = lambda L: np.concatenate(L).astype(np.float64)
        return cat(X), np.concatenate(y), cat(Xd), cat(Xm)
    Xtr, ytr, Xdtr, Mtr = zfeat(DA, trA); Xte, yte, Xdte, Mte = zfeat(DA, teA)
    ex["absorption_probe_auroc"] = _logreg_auroc(Xtr, ytr, Xte, yte, auroc)
    ex["absorption_probe_auroc_dfeat"] = _logreg_auroc(Xdtr, ytr, Xdte, yte, auroc)
    # revision 2: regress z_t on arm and gripper motion (fitted on training steps) and probe only the residual
    mu, sd = Mtr.mean(0), Mtr.std(0) + 1e-8
    P = lambda M: np.c_[(M - mu) / sd, np.ones(len(M))]
    W = np.linalg.solve(P(Mtr).T @ P(Mtr) + 1e-3 * np.eye(P(Mtr).shape[1]), P(Mtr).T @ Xtr)
    ex["absorption_probe_auroc_resid"] = _logreg_auroc(Xtr - P(Mtr) @ W, ytr, Xte - P(Mte) @ W, yte, auroc)
    ex["absorption_probe_auroc_motion"] = _logreg_auroc(P(Mtr)[:, :-1], ytr, P(Mte)[:, :-1], yte, auroc)

    # ------------------------------------------------ 7. few-shot with the drift generator also fine-tuned
    if has_drift and kind != "oracle":
        base = copy.deepcopy(model.state_dict()); o0b, o1b, _ = g["pairs"](DB, teB); out = {}
        for k in (0, 1, 3, 5):
            model.load_state_dict(base)
            if k:
                a0, a1, _ = g["pairs"](DB, trB[:k])
                opt = torch.optim.Adam(list(model.idm_net.parameters()) + [model.S0, model.L0, model.b0], lr=1e-3)
                for _ in range(args.fewshot_steps):
                    i = torch.randint(0, len(a0), (256,), device=a0.device)
                    with torch.no_grad():
                        s0, s1t = model.encode(a0[i]), model.encode(a1[i])
                    s1h = model.fwd(s0, model.idm(a0[i], a1[i]))
                    l = mse(model.decode(s1h), a1[i]) + mse(s1h, s1t)
                    opt.zero_grad(); l.backward(); opt.step()
            with torch.no_grad():
                pred = model.decode(model.fwd(model.encode(o0b), model.idm(o0b, o1b)))
                out[k] = (mse(pred, o1b) / mse(o0b, o1b)).item()
        model.load_state_dict(base)
        ex["fewshot_fwd_rel_err_tuneG0"] = out

    res.update(ex)


def _roll(model, s, zs):
    for z in zs:
        s = model.fwd(s, z)
    return s
