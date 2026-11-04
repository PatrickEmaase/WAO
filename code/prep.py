"""Decode cam_high at 10 Hz, extract DINOv2 ViT-S/14 features, save per-direction npz."""
import glob, h5py, cv2, numpy as np, torch, torch.nn.functional as F

dev = "cuda"
model = torch.hub.load("/home/pemaase/.cache/torch/hub/facebookresearch_dinov2_main", "dinov2_vits14", source="local").to(dev).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
SUB = 5                                    # 50 Hz -> 10 Hz

@torch.no_grad()
def feats(frames):                         # list of BGR uint8 (480, 848, 3)
    x = torch.from_numpy(np.stack([f[:, :, ::-1] for f in frames]).copy()).to(dev).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(224, 392), mode="bilinear", antialias=True)
    out = model.forward_features((x - mean) / std)
    cls, pt = out["x_norm_clstoken"], out["x_norm_patchtokens"]          # (B,384), (B,16*28,384)
    grid = F.adaptive_avg_pool2d(pt.view(-1, 16, 28, 384).permute(0, 3, 1, 2), (2, 4)).flatten(1)  # (B, 3072)
    return cls.cpu().numpy(), pt.mean(1).cpu().numpy(), grid.half().cpu().numpy()

for d in ["toaster_forward", "toaster_backward"]:
    files = sorted(glob.glob(f"/home/pemaase/patrick/{d}/*.hdf5"), key=lambda f: int(f.split("_")[-1][:-5]))
    C, M, G, Q, SP, IDS = [], [], [], [], [], []
    for i, p in enumerate(files):
        with h5py.File(p) as f:
            idx = np.arange(0, f["action"].shape[0], SUB)
            frames = [cv2.imdecode(f["observations/images/cam_high"][t], 1) for t in idx]
            q = f["observations/qpos"][:]
        c, m, g = feats(frames)
        arm = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        speed = np.abs(np.diff(q[:, arm], axis=0, append=q[-1:, arm])).sum(1)       # per raw step
        speed10 = speed[: len(idx) * SUB].reshape(len(idx), SUB).mean(1)
        C.append(c); M.append(m); G.append(g); Q.append(q[idx]); SP.append(speed10); IDS.append(int(p.split("_")[-1][:-5]))
        print(d, i, flush=True)
    np.savez(f"/home/pemaase/patrick/rave/{d}.npz", cls=np.stack(C), mean=np.stack(M), grid=np.stack(G),
             qpos=np.stack(Q), speed=np.stack(SP), ids=np.array(IDS))
