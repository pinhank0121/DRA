"""
Linear probe on pooled LightningDiT features for ImageNet, used for the ImageNet
frequency-saliency analysis (App. B, Fig. 9).

Phase 1: extract pooled layer-14 features (t=0.8) for a stride-subset of the cached
         train latents (ImgLatentDataset handles per-source normalization).
Phase 2: train a 1000-way linear probe on the cached features.

Output: <out-dir>/probe.pt  {"weight","bias","train_acc","feat_dim"}
        <out-dir>/feats.npy / labels.npy (fp16 cache, reusable)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latent-dir", default="data/imagenet_vae_latents_bf16/vavae_f16d32/imagenet_train_256")
    p.add_argument("--stride", type=int, default=6, help="~1.28M/6 = 213k probe-training samples")
    p.add_argument("--out-dir", default="checkpoints/202607_neurips/ldit_probe_in1k")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    feats_path = os.path.join(args.out_dir, "feats.npy")
    labels_path = os.path.join(args.out_dir, "labels.npy")

    if not os.path.exists(feats_path):
        from core.data.img_latent_pair import ImgLatentDataset
        from core.models.lightning_dit.feature_extractor import LightningDiTFeatureExtractor
        ds = ImgLatentDataset(args.latent_dir, latent_norm=True, flip_latents=False)
        idx = list(range(0, len(ds), args.stride))
        sub = torch.utils.data.Subset(ds, idx)
        dl = DataLoader(sub, batch_size=args.batch_size, num_workers=8, shuffle=False)
        ldit = LightningDiTFeatureExtractor(
            config_path="LightningDiT/configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml",
            checkpoint_path="checkpoints/ldit_checkpoints/ldit-xl-img256-800ep.pt",
            feature_layers=[14], timestep=0.8, device=device, freeze=True)
        n = len(sub)
        feats = np.lib.format.open_memmap(feats_path, mode="w+", dtype=np.float16, shape=(n, 1152))
        labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.int64, shape=(n,))
        pos = 0
        with torch.no_grad():
            for lat, lab in dl:
                f = ldit(lat.to(device))["features"][0].float().mean(dim=1)  # (B, 1152)
                feats[pos:pos + len(lab)] = f.to(torch.float16).cpu().numpy()
                labels[pos:pos + len(lab)] = lab.numpy()
                pos += len(lab)
                if pos % (args.batch_size * 100) < args.batch_size:
                    print(f"extract {pos}/{n}", flush=True)
        feats.flush(); labels.flush()
        print("extraction done:", n)

    X = torch.from_numpy(np.asarray(np.lib.format.open_memmap(feats_path, mode="r"))).float()
    y = torch.from_numpy(np.asarray(np.lib.format.open_memmap(labels_path, mode="r")))
    n = len(y)
    print("probe training on", n, "samples")
    probe = torch.nn.Linear(1152, 1000).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        correct = 0
        for i in range(0, n, 4096):
            b = perm[i:i + 4096]
            xb, yb = X[b].to(device), y[b].to(device)
            logits = probe(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
        sched.step()
        print(f"epoch {ep + 1}/{args.epochs} train acc {correct / n:.4f}", flush=True)
    torch.save({"weight": probe.weight.detach().cpu(), "bias": probe.bias.detach().cpu(),
                "train_acc": correct / n, "feat_dim": 1152}, os.path.join(args.out_dir, "probe.pt"))
    print("probe saved")


if __name__ == "__main__":
    main()
