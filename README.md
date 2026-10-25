# Expanding the Role of Diffusion Models for Robust Classifier Training

This repository contains the reference implementation of **diffusion representation-alignment (DRA)
regularizer** on top of DM-AT adversarial training recipe, using features extracted from
a pretrained diffusion model (EDM / LightningDiT) as the alignment target.
It supports CIFAR-10/100 and ImageNet-1k.

The training/AT scaffolding builds on
[DM-Improves-AT](https://github.com/wzekai99/DM-Improves-AT) (Wang et al.,
ICML 2023), and the ImageNet generative side relies on
[LightningDiT](https://github.com/hustvl/LightningDiT) (Yao et al., CVPR 2025).
A bundled, lightly modified copy of LightningDiT is shipped under
[`LightningDiT/`](./LightningDiT) so that image sampling and VAE latent
extraction work out of the box.

A complete, runnable copy of every pipeline below is also available in
[`run_scripts/run.sh`](./run_scripts/run.sh).

---

## 1. Environment

The project is tested with python 3.12.8, torch 2.6.0 (cuda 12.4), torchvision 0.21.0, and Nvidia H100 GPU.

The core DM-AT environment needs the standard deep-learning stack: `torch`,
`torchvision`, `timm`, `einops`, `accelerate`, `lion-pytorch`, plus utilities
such as `numpy`, `pyyaml`, `tqdm`. The optional Stable-Diffusion feature
precompute scripts (`train_scripts/precompute-cifar-sd-features*.py`) additionally
need `diffusers` and `transformers`.

In addition, install AutoAttack and RandAugment from source:

```bash
pip install git+https://github.com/fra31/auto-attack
pip install git+https://github.com/ildoonet/pytorch-randaugment
```

LightningDiT (used for ImageNet image generation and VAE latent extraction)
uses its own environment — see
[`LightningDiT/README.md`](./LightningDiT/README.md) for the exact package
list.

---

## 2. Repository layout

```
.
├── train_scripts/         # training entry points
│   ├── train-wa-repa-pure.py    # CIFAR-10, no extra data
│   ├── train-wa-repa.py         # CIFAR-10 + EDM 50M
│   ├── train-wa-repa-in1k.py    # ImageNet-1k (+ optional 4M aux data)
│   ├── precompute-cifar-*.py    # optional: caches for foundation-teacher ablations (§7)
│   └── train-ldit-probe-in1k.py # optional: linear probe on LightningDiT features (§7)
├── eval_scripts/          # AutoAttack / RobustBench evaluation
│   ├── eval-aa.py
│   └── eval-last-aa.py
├── core/                  # AT framework (data, attacks, training utils)
│   └── models/            # WideResNet / ViT / DinoV3 backbones with REPA heads
├── run_scripts/run.sh     # one-shot reference commands for every config
├── LightningDiT/          # bundled LightningDiT for IN1k generation + latents
└── data/                  # symlink target for datasets (see §4)
```

`data/` is expected to be a directory or symlink that contains the raw
datasets, EDM `.npz` files, the optional ImageNet auxiliary set and pre-extracted VAE latents (see §3 below).

---

## 3. Data and checkpoints

All training/eval commands assume a single top-level `data/` directory and a
`checkpoints/` directory at the repo root. The simplest setup is:

```bash
mkdir -p data checkpoints checkpoints/ldit_checkpoints
# or, if the datasets live on a separate disk:
ln -s /path/to/your/datasets data
```

### 3a. CIFAR-10 / CIFAR-100 (raw)

The torchvision-format CIFAR-10/100 datasets are downloaded automatically by
the training scripts on first run; nothing to prepare.

### 3b. EDM-generated auxiliary data (CIFAR-10/100)

The EDM-generated `.npz` shards are hosted on the HuggingFace release of the
original DM-Improves-AT project:

- CIFAR-10: <https://huggingface.co/datasets/P2333/DM-Improves-AT/tree/main/cifar10>
  (1M / 5M / 10M / 20M / 50M)
- CIFAR-100: <https://huggingface.co/datasets/P2333/DM-Improves-AT/tree/main/cifar100>

Download the shard(s) you need into `data/`. The 20M and 50M sets are split
across multiple files; merge them with [`merge_data.py`](./merge_data.py) so
you end up with, e.g.:

```
data/1m.npz
data/50m.npz
```

The CIFAR-10 50M training command in §4b expects `data/50m.npz`.

### 3c. ImageNet-1k (raw)

Place the standard ImageNet-1k train/val splits under
`data/imagenet/{train,val}/<wnid>/*.JPEG`. The LightningDiT extraction
scripts in §5b read from these paths.

### 3d. LightningDiT checkpoint

Download the LightningDiT XL/16 + VA-VAE f16d32 800-epoch checkpoint
(`ldit-xl-img256-800ep.pt`) from the LightningDiT release page
(<https://github.com/hustvl/LightningDiT#-pre-trained-models>) and place it at:

```
checkpoints/ldit_checkpoints/ldit-xl-img256-800ep.pt
```

This same checkpoint is used both by the LightningDiT generation /
latent-extraction pipeline and as the DRA alignment target during ImageNet
adversarial training.

### 3e. ImageNet auxiliary set + pre-computed VAE latents

These are produced by the LightningDiT pipeline described in §5a–5b and end
up under:

```
data/imagenet_ldit4M/                                              # generated images
data/imagenet_vae_latents_bf16/vavae_f16d32/imagenet_train_256/    # latents for real train
data/imagenet_vae_latents_bf16/vavae_f16d32/imagenet_ldit4M_256/   # latents for generated set
```

If you only want to train without the auxiliary set, you can skip §5a and
extract latents only for `imagenet_train`.

---

## 4. CIFAR-10

### 4a. CIFAR-10, no extra data (WRN-34-10 + DRA)

```bash
python -m train_scripts.train-wa-repa-pure --data-dir 'data' \
  --log-dir 'checkpoints/cifar10_0M/' \
  --desc 'PoolRepa6_WRN34-10Swish-8x8_cifar10_lr0p2_TRADES5_epoch200_bs512_ls0p1' \
  --data cifar10 \
  --batch-size 512 --batch-size-validation 1024 \
  --model wrn-34-10-swish-repa-8x8-pool \
  --num-adv-epochs 200 \
  --lr 0.2 --beta 5.0 --feat-reg-coeff 6.0 \
  --ls 0.1 --sigma 0.1 --perturb \
  --feature_layer_name "8x8_block4" \
  --accumulation_steps 1 --adv-eval-freq 1
```

`--feat-reg-coeff` is the DRA weight (= λ · β; we use λ = 1.2 by default).
Setting `--feat-reg-coeff 0.0` reproduces the plain TRADES baseline.

Evaluation under AutoAttack (ℓ∞, ε = 8/255):

```bash
python -m eval_scripts.eval-aa --data-dir 'data' \
  --log-dir 'checkpoints/cifar10_0M/' \
  --desc  'PoolRepa6_WRN34-10Swish-8x8_cifar10_lr0p2_TRADES5_epoch200_bs512_ls0p1' \
  --eps 0.031372549
```

### 4b. CIFAR-10 + EDM 50M (ViT-B/16 + DRA)

Assumes you have already prepared `data/50m.npz` per §3b.

```bash
python -m train_scripts.train-wa-repa --data-dir 'data' \
  --log-dir 'checkpoints/cifar10_50M' \
  --desc 'PoolRepa0_vit-b-repa-pool_cifar10_lr1e-4_TRADES3_epoch2000_bs1024_ls0p4_lion_wd0p5' \
  --data cifar10s \
  --batch-size 1024 --batch-size-validation 1024 \
  --model vit-b-repa-pool \
  --num-adv-epochs 2000 --adv-eval-freq 25 \
  --lr 1e-4 --weight-decay 0.5 --beta 3.0 \
  --opt-type lion \
  --unsup-fraction 0.9 \
  --aux-data-filename 'data/50m.npz' \
  --ls 0.4 --sigma 0.1 \
  --feat-reg-coeff 0.0 \
  --dual-time --perturb \
  --feature_layer_name "8x8_block4" \
  --accumulation_steps 1
```

Evaluation:

```bash
python -m eval_scripts.eval-last-aa --data-dir 'data' \
  --log-dir 'checkpoints/cifar10_50M' \
  --desc 'PoolRepa0_vit-b-repa-pool_cifar10_lr1e-4_TRADES3_epoch2000_bs1024_ls0p4_lion_wd0p5' \
  --eps 0.031372549
```

---

## 5. ImageNet-1k

The ImageNet pipeline has three stages:

1. **(Optional) Generate auxiliary images** with LightningDiT, producing the
   `imagenet_ldit4M` set used as TRADES auxiliary data.
2. **Extract VAE latents** for both real ImageNet train images and the
   generated auxiliary set. VAE latents are consumed directly by the DRA
   regularizer at training time, avoiding repeated VAE encodes.
3. **Adversarial training** with DRA against LightningDiT features.

### 5a. Generate auxiliary images (`run_inference.sh`)

Make sure the LightningDiT checkpoint is in place per §3d, then sample:

```bash
cd LightningDiT
GPUS_PER_NODE=8 bash run_inference.sh \
    configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml
```

Configure the desired number of samples / class-conditional setup in the YAML
config. The output directory will become the `data/imagenet_ldit4M/` set
referenced below; rename or symlink accordingly.

### 5b. Extract VAE latents (`run_extraction.sh`)

`run_extraction.sh` runs `extract_features.py` and writes bf16 latent shards to
`data/imagenet_vae_latents_bf16/vavae_f16d32/<split>/`. The script ships with
three variants (real `imagenet_val`, real `imagenet_train`, generated
`imagenet_ldit4M`); uncomment the ones you need.

```bash
cd LightningDiT
GPUS_PER_NODE=8 bash run_extraction.sh \
    configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml
```

You should end up with at least:

```
data/imagenet_vae_latents_bf16/vavae_f16d32/imagenet_train_256/
data/imagenet_vae_latents_bf16/vavae_f16d32/imagenet_ldit4M_256/   # if using aux data
```

### 5c. Adversarial training with DRA on ImageNet-1k

`NUM_GPUS` should match `--nproc_per_node`. Example: 4 GPUs, ViT-B/16 with
DinoV3 init, ℓ∞ ε = 4/255, REPA against LightningDiT layer 14 features
(timestep 0.8):

```bash
MASTER_PORT=$((RANDOM % 10000 + 20000))
TIMM_FUSED_ATTN=2 OMP_NUM_THREADS=1 torchrun \
    --nproc_per_node=4 --master_port=$MASTER_PORT \
    -m train_scripts.train-wa-repa-in1k \
    --data-dir 'data' \
    --log-dir 'checkpoints/imagenet_ldit4M/pre-vit-b-dinov3-baseaug' \
    --desc  'PoolRepa12_pre-vit-b-dinov3_IN1k_lr1e-4_TRADES10_epoch100_bs1024_ls0p1_lion_wd0p0_clipgrad1p0' \
    --data imagenet --augment base \
    --batch-size 1024 --batch-size-validation 1024 \
    --model pre-vit-b-dinov3-repa-pool \
    --num-adv-epochs 100 \
    --opt-type lion --lr 1e-4 --weight-decay 0.0 \
    --beta 10.0 --feat-reg-coeff 12.0 \
    --use-latent-repa \
    --lightning-dit-config LightningDiT/configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml \
    --lightning-dit-ckpt   checkpoints/ldit_checkpoints/ldit-xl-img256-800ep.pt \
    --latent-dir       data/imagenet_vae_latents_bf16/vavae_f16d32/imagenet_train_256 \
    --aux-data-filename data/imagenet_ldit4M \
    --aux-latent-dir   data/imagenet_vae_latents_bf16/vavae_f16d32/imagenet_ldit4M_256 \
    --dit-feature-layers 14 --dit-timestep 0.8 \
    --num-timesteps 2 --mix-fixed-sampled-timesteps \
    --repa_dim 1152 \
    --ls 0.1 --clip-grad 1.0 --accumulation_steps 1 \
    --attack-iter 3 --attack-step "8/(255*3)" --attack-eps 4/255 \
    --unsup-fraction 0.8 --adv-eval-freq 1
```

Drop `--aux-data-filename` / `--aux-latent-dir` (and set
`--unsup-fraction 0`) to train without the LightningDiT-generated 4M set.

### 5d. ImageNet evaluation

```bash
python -m eval_scripts.eval-last-aa --data-dir 'data' \
  --log-dir 'checkpoints/imagenet_ldit4M/pre-vit-b-dinov3-baseaug' \
  --desc   'PoolRepa12_pre-vit-b-dinov3_IN1k_lr1e-4_TRADES10_epoch100_bs1024_ls0p1_lion_wd0p0_clipgrad1p0' \
  --eps 0.0156862745
```

For ℓ2 add `--aa-norm L2 --eps 2.0`.

---

## 6. Key hyper-parameters

| flag                                               | meaning                                                                                                                           |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `--beta`                                           | weight value beta for TRADES (AT loss used in DM-AT)                                                                              |
| `--feat-reg-coeff`                                 | DRA loss weight (= λ · β); 0 disables DRA                                                                                         |
| `--repa_dim`                                       | projection dimension of the DRA head, should match the target dimension (256 for CIFAR-10 EDM and 1152 for ImageNet LightningDiT) |
| `--feature_layer_name`                             | CIFAR-10/100: which EDM diffusion layer to align with (e.g. `8x8_block4`)                                                         |
| `--perturb`, `--sigma`                             | CIFAR-10/100: input noise injection during diffusion feature extraction                                                           |
| `--dual-time`                                      | CIFAR-10/100: DRA, sample an additional timestep for alignment target, used for exp using more than 20M synthetic images          |
| `--use-latent-repa`                                | ImageNet: align against LightningDiT latent features (IN1k)                                                                       |
| `--dit-feature-layers`                             | ImageNet: which DiT block(s) to use as the alignment target                                                                       |
| `--dit-timestep`                                   | ImageNet: diffusion timestep at which DiT features are extracted                                                                  |
| `--num-timesteps`, `--mix-fixed-sampled-timesteps` | ImageNet: sampling strategy for DiT timesteps                                                                                     |

---

## 7. Optional diagnostics and ablation utilities

None of the following is needed to reproduce the main results; they support the
ablations and diagnostics reported in the paper's appendix. Defaults
leave all of them switched off.

**Training / evaluation flags**

| flag                                              | meaning                                                                                                                                                  |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--feat-reg-warmup-epochs N`                      | linearly ramp `--feat-reg-coeff` from 0 to its full value over the first N epochs (0 = off)                                                             |
| `--log-grad-cosine`, `--grad-cosine-interval N`   | log cos(∇L_AT, ∇L_align) every N batches (teacher-REPA path; costs two extra backward passes)                                                            |
| `--teacher-feat-standardize`                      | ImageNet SSL teachers: per-channel standardization of the target features                                                                               |
| `--opt-type muon`                                 | Muon optimizer (`timm.optim.Muon`; AdamW fallback for embeddings / cls token / head)                                                                     |
| `--model frozen-edm-athead`                       | CIFAR: frozen unconditional EDM encoder + trainable head, trained adversarially. Use with `--feat-reg-coeff 0`                                            |
| `--use-latent-repa` (CIFAR)                       | CIFAR student aligned to the off-the-shelf ImageNet LightningDiT, reading cached VA-VAE latents from `--latent-dir` (see below)                           |
| `--cached-feature-teacher`                        | CIFAR: alignment targets are pre-computed pooled feature vectors in `--latent-dir` (e.g. Stable-Diffusion features); set `--repa_dim` to their dimension |
| `--attacks-to-run`, `--aa-num-samples` (eval)     | `eval-last-aa.py`: run a subset of AutoAttack components / evaluate only the first N test images                                                        |

**Cache-building scripts** (`python -m train_scripts.<name> --shard g --num-shards G --out-dir <dir> [--init]`,
one process per GPU; run `--init` once on shard 0 first)

| script                                             | produces                                                                                                                                                                 |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `train_scripts/precompute-cifar-vae-latents.py`     | VA-VAE f16d32 latents of CIFAR-10 train + `1m.npz` (upsampled to 256) for `--use-latent-repa` on CIFAR; needs `LightningDiT/tokenizer/configs/vavae_f16d32.yaml` + VA-VAE ckpt |
| `train_scripts/precompute-cifar-sd-features.py`     | pooled Stable-Diffusion-1.5 mid-block features (empty prompt) for `--cached-feature-teacher`; needs `diffusers`, `transformers`                                        |
| `train_scripts/precompute-cifar-sd-features-clstext.py` | same, conditioned on `"a photo of a {class}"` prompts                                                                                                              |
| `train_scripts/train-ldit-probe-in1k.py`            | 1000-way linear probe on pooled LightningDiT layer-14 features of the cached ImageNet latents (`probe.pt`), used for the saliency analysis                              |

The cache index space is `[0:50000]` = CIFAR-10 train order followed by the
`1m.npz` order, matching the semi-supervised dataset; do not combine with
`--aux-take-amount`.

---

## 8. Acknowledgements

This codebase builds directly on:

- [DM-Improves-AT](https://github.com/wzekai99/DM-Improves-AT) (ICML 2023) for
  the AT framework, EDM-augmented training recipe, and CIFAR-10 baselines.
- [LightningDiT](https://github.com/hustvl/LightningDiT) (CVPR 2025)
  for the bundled latent diffusion transformer, VAE, image sampling, and VAE
  latent extraction utilities under [`LightningDiT/`](./LightningDiT).
- [AutoAttack](https://github.com/fra31/auto-attack) and
  [RobustBench](https://robustbench.github.io) for evaluation.

We thank the authors of these projects for releasing high-quality, reusable
code. License notices for the bundled third-party components are listed in
[`NOTICE.md`](./NOTICE.md).

---

## 9. Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{huang2026expanding,
  title={Expanding the Role of Diffusion Models for Robust Classifier Training},
  author={Pin-Han Huang and Shang-Tse Chen and Hsuan-Tien Lin},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026},
}
```
