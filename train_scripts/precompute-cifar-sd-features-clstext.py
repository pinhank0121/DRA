"""
Reactive follow-up to the SD-1.5 off-the-shelf rung: identical protocol to
precompute-cifar-sd-features.py, but the UNet is conditioned on the class-name
text prompt "a photo of a {class}" (CIFAR-10 label -> CLIP embedding) instead of
the empty prompt. Everything else (VAE, t=100, mid-block pooled 1280-d, fp16,
index space, sharding) is unchanged, so cond-vs-uncond is a clean single-factor
ablation of the text input.

Run one process per GPU:
  CUDA_VISIBLE_DEVICES=g python -m train_scripts.precompute-cifar-sd-features-clstext \
      --shard g --num-shards 4 --out-dir <dir> [--init]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
N_TRAIN, N_AUX = 50000, 1000000
FEAT_DIM = 1280
T_STEP = 100
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def load_images():
    import torchvision
    cifar = torchvision.datasets.CIFAR10(os.path.join(REPO, "data/cifar10"), train=True, download=False)
    aux = np.load(os.path.join(REPO, "data/edm_data/cifar10/1m.npz"))
    labels = np.concatenate([np.asarray(cifar.targets, dtype=np.int64), aux["label"].astype(np.int64)])
    return cifar.data, aux["image"], labels


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
        prompts = [f"a photo of a {c}" for c in CLASS_NAMES]
        ids = tok(prompts, padding="max_length", max_length=tok.model_max_length, return_tensors="pt")
        cls_embs = te(ids.input_ids.to(device))[0]  # (10, 77, 768)

    feats_out = {}
    hook = unet.mid_block.register_forward_hook(lambda m, i, o: feats_out.__setitem__("mid", o))

    alphas = sched.alphas_cumprod.to(device=device, dtype=dtype)
    t = torch.tensor([T_STEP], device=device)
    sqrt_a, sqrt_1ma = alphas[T_STEP].sqrt(), (1 - alphas[T_STEP]).sqrt()

    mm = {k: np.lib.format.open_memmap(p, mode="r+") for k, p in paths.items()}
    cifar_data, aux, labels = load_images()
    per = int(np.ceil(total / args.num_shards))
    lo0, hi0 = args.shard * per, min((args.shard + 1) * per, total)
    print(f"shard {args.shard}: [{lo0}, {hi0})", flush=True)
    torch.manual_seed(4321 + args.shard)

    for lo in range(lo0, hi0, args.batch_size):
        hi = min(lo + args.batch_size, hi0)
        imgs = get_batch(cifar_data, aux, lo, hi)
        emb = cls_embs[torch.from_numpy(labels[lo:hi]).to(device)]  # (B, 77, 768)
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device=device, dtype=dtype).div_(255.0)
        x = F.interpolate(x, size=512, mode="bicubic", align_corners=False).clamp_(0, 1).sub_(0.5).div_(0.5)
        with torch.no_grad():
            for key, xx in (("latents", x), ("latents_flip", torch.flip(x, dims=[3]))):
                lat = vae.encode(xx).latent_dist.sample() * 0.18215
                noisy = sqrt_a * lat + sqrt_1ma * torch.randn_like(lat)
                _ = unet(noisy, t.expand(noisy.shape[0]), encoder_hidden_states=emb)
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
