"""
Off-the-shelf diffusion teacher (App. J): precompute pooled Stable-Diffusion-1.5 mid-block features for the
CIFAR-10 semisup set (50k train + 1M synthetic), so a literal text-to-image foundation
model (never trained on CIFAR) serves as the DRA alignment teacher.

Protocol (frozen off-the-shelf, DIFT/SD-DINO style):
  32 -> 512 bicubic -> SD VAE encode (posterior.sample() * 0.18215, one fixed draw)
  -> add scheduler noise at fixed t=100/1000 (low-noise plateau)
  -> UNet forward with the empty-prompt ("") CLIP embedding (unconditional)
  -> mid_block output (1280, 8, 8) -> spatial mean -> (1280,) fp16.

Index space matches SemiSupervisedDataset: [0:50000] CIFAR train order, then 1m.npz.
Outputs: latents.npy (N,1280), latents_flip.npy, shard{i}.done  (no latents_stats.pt —
cosine alignment is scale-invariant, the training path skips normalization).

Run one process per GPU:
  CUDA_VISIBLE_DEVICES=g python -m train_scripts.precompute-cifar-sd-features \
      --shard g --num-shards 4 --out-dir <dir> [--init]
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
N_TRAIN, N_AUX = 50000, 1000000
FEAT_DIM = 1280
T_STEP = 100


def load_images():
    import torchvision
    cifar = torchvision.datasets.CIFAR10(os.path.join(REPO, "data/cifar10"), train=True, download=False)
    aux = np.load(os.path.join(REPO, "data/edm_data/cifar10/1m.npz"))["image"]
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--init", action="store_true")
    args = parser.parse_args()

    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    total = N_TRAIN + N_AUX
    paths = {k: os.path.join(args.out_dir, f"{k}.npy") for k in ("latents", "latents_flip")}
    if args.init:
        for p in paths.values():
            if not os.path.exists(p):
                np.lib.format.open_memmap(p, mode="w+", dtype=np.float16, shape=(total, FEAT_DIM))
        print("memmaps allocated")

    done_marker = os.path.join(args.out_dir, f"shard{args.shard}.done")
    if os.path.exists(done_marker):
        print(f"shard {args.shard} already done")
        return

    device = torch.device("cuda")
    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=dtype).to(device).eval()
    sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    tok = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()

    with torch.no_grad():
        ids = tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt")
        empty_emb = te(ids.input_ids.to(device))[0]  # (1, 77, 768)

    feats_out = {}
    hook = unet.mid_block.register_forward_hook(lambda m, i, o: feats_out.__setitem__("mid", o))

    alphas = sched.alphas_cumprod.to(device=device, dtype=dtype)
    t = torch.tensor([T_STEP], device=device)
    sqrt_a, sqrt_1ma = alphas[T_STEP].sqrt(), (1 - alphas[T_STEP]).sqrt()

    mm = {k: np.lib.format.open_memmap(p, mode="r+") for k, p in paths.items()}
    cifar_data, aux = load_images()
    per = int(np.ceil(total / args.num_shards))
    lo0, hi0 = args.shard * per, min((args.shard + 1) * per, total)
    print(f"shard {args.shard}: [{lo0}, {hi0})", flush=True)
    torch.manual_seed(4321 + args.shard)

    for lo in range(lo0, hi0, args.batch_size):
        hi = min(lo + args.batch_size, hi0)
        imgs = get_batch(cifar_data, aux, lo, hi)
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device=device, dtype=dtype).div_(255.0)
        x = F.interpolate(x, size=512, mode="bicubic", align_corners=False).clamp_(0, 1).sub_(0.5).div_(0.5)
        with torch.no_grad():
            for key, xx in (("latents", x), ("latents_flip", torch.flip(x, dims=[3]))):
                lat = vae.encode(xx).latent_dist.sample() * 0.18215
                noisy = sqrt_a * lat + sqrt_1ma * torch.randn_like(lat)
                _ = unet(noisy, t.expand(noisy.shape[0]), encoder_hidden_states=empty_emb.expand(noisy.shape[0], -1, -1))
                mid = feats_out["mid"]  # (B, 1280, 8, 8)
                mm[key][lo:hi] = mid.float().mean(dim=[2, 3]).to(torch.float16).cpu().numpy()
        if (lo - lo0) % (args.batch_size * 50) == 0:
            print(f"shard {args.shard}: {lo - lo0}/{hi0 - lo0}", flush=True)

    for m in mm.values():
        m.flush()
    hook.remove()
    with open(done_marker, "w") as f:
        f.write("done\n")
    print(f"shard {args.shard} complete")


if __name__ == "__main__":
    main()
