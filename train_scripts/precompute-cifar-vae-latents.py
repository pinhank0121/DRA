"""
Off-the-shelf diffusion teacher (cf. App. J): precompute VA-VAE f16d32 latents for the
CIFAR-10 semisup training set (50k torchvision-order train + 1M EDM synthetic in
npz order), upsampled 32->256 bicubic, so the off-the-shelf ImageNet LightningDiT
teacher can be used through the existing latent-REPA path.

Index space matches SemiSupervisedDataset exactly: [0:50000] = CIFAR-10 train,
[50000:1050000] = 1m.npz 'image' order.

Outputs in --out-dir:
  latents.npy       (1050000, 32, 16, 16) fp16  — unflipped
  latents_flip.npy  same, hflipped input        — mirrors in1k latents_flip
  latents_stats.pt  per-channel mean/std of unflipped (in1k protocol)
  shard{i}.done     resume markers

Run one process per GPU:
  CUDA_VISIBLE_DEVICES=g python -m train_scripts.precompute-cifar-vae-latents \
      --shard g --num-shards 4 --out-dir <dir> [--init]
(--init on shard 0 first, alone, to allocate the memmaps.)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "LightningDiT"))

VAE_CONFIG = os.path.join(REPO, "LightningDiT/tokenizer/configs/vavae_f16d32.yaml")
N_TRAIN, N_AUX = 50000, 1000000
LATENT_SHAPE = (32, 16, 16)


def load_images():
    import torchvision
    cifar = torchvision.datasets.CIFAR10(os.path.join(REPO, "data/cifar10"), train=True, download=False)
    aux = np.load(os.path.join(REPO, "data/edm_data/cifar10/1m.npz"))["image"]
    assert cifar.data.shape == (N_TRAIN, 32, 32, 3) and aux.shape == (N_AUX, 32, 32, 3)
    return cifar.data, aux


def get_batch(cifar_data, aux, lo, hi):
    idx = np.arange(lo, hi)
    imgs = np.empty((len(idx), 32, 32, 3), dtype=np.uint8)
    base = idx < N_TRAIN
    imgs[base] = cifar_data[idx[base]]
    imgs[~base] = aux[idx[~base] - N_TRAIN]
    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--init", action="store_true", help="allocate memmap files (run once, alone)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total = N_TRAIN + N_AUX
    paths = {k: os.path.join(args.out_dir, f"{k}.npy") for k in ("latents", "latents_flip")}

    if args.init:
        for p in paths.values():
            if not os.path.exists(p):
                np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(total,) + LATENT_SHAPE)
        print("memmaps allocated")

    done_marker = os.path.join(args.out_dir, f"shard{args.shard}.done")
    if os.path.exists(done_marker):
        print(f"shard {args.shard} already done")
        return

    # VA_VAE config has a cwd-relative ckpt_path ("ldit_checkpoints/...") — chdir to
    # checkpoints/ so it resolves, matching how extract_features.py was run.
    os.chdir(os.path.join(REPO, "checkpoints"))
    from tokenizer.vavae import VA_VAE
    vae = VA_VAE(VAE_CONFIG)

    mm = {k: np.lib.format.open_memmap(p, mode="r+") for k, p in paths.items()}
    cifar_data, aux = load_images()

    per = int(np.ceil(total / args.num_shards))
    lo0, hi0 = args.shard * per, min((args.shard + 1) * per, total)
    print(f"shard {args.shard}: [{lo0}, {hi0})")

    torch.manual_seed(1234 + args.shard)  # VAE posterior.sample() noise
    for lo in range(lo0, hi0, args.batch_size):
        hi = min(lo + args.batch_size, hi0)
        imgs = get_batch(cifar_data, aux, lo, hi)
        x = torch.from_numpy(imgs).float().permute(0, 3, 1, 2).div_(255.0)
        x = F.interpolate(x, size=256, mode="bicubic", align_corners=False).clamp_(0, 1)
        x = x.sub_(0.5).div_(0.5)
        for key, xx in (("latents", x), ("latents_flip", torch.flip(x, dims=[3]))):
            z = vae.encode_images(xx)
            mm[key][lo:hi] = z.to(torch.float16).cpu().numpy()
        if (lo - lo0) % (args.batch_size * 50) == 0:
            print(f"shard {args.shard}: {lo - lo0}/{hi0 - lo0}", flush=True)

    for m in mm.values():
        m.flush()
    with open(done_marker, "w") as f:
        f.write("done\n")
    print(f"shard {args.shard} complete")

    # Shard 0 computes stats once all shards are done (in1k per-source protocol).
    if args.shard == 0:
        import time
        while not all(os.path.exists(os.path.join(args.out_dir, f"shard{i}.done")) for i in range(args.num_shards)):
            time.sleep(30)
        rng = np.random.default_rng(0)
        sample = rng.choice(total, 10000, replace=False)
        feats = torch.from_numpy(np.asarray(mm["latents"][np.sort(sample)])).to(torch.float64)
        mean = feats.mean(dim=[0, 2, 3], keepdim=True)[0:1]
        std = feats.std(dim=[0, 2, 3], keepdim=True)[0:1]
        torch.save({"mean": mean.float(), "std": std.float()}, os.path.join(args.out_dir, "latents_stats.pt"))
        print("stats saved:", mean.flatten()[:4], std.flatten()[:4])


if __name__ == "__main__":
    main()
