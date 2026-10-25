#!/bin/bash

# feat-reg-coeff = lambda * beta (Trades Coefficient)
# lambda is by default 1.2 in our experiments
# set feat-reg-coeff to 0.0 for baseline

# CIFAR-10
python -m train_scripts.train-wa-repa-pure --data-dir 'data' \
  --log-dir 'checkpoints/202604/cifar10_0M/' \
  --desc 'PoolRepa6_WRN34-10Swish-8x8_cifar10_lr0p2_TRADES5_epoch200_bs512_ls0p1/s2' \
  --data cifar10 \
  --batch-size 512 \
  --batch-size-validation 1024 \
  --model wrn-34-10-swish-repa-8x8-pool \
  --num-adv-epochs 200 \
  --lr 0.2 \
  --beta 5.0 \
  --feat-reg-coeff 6.0 \
  --ls 0.1 \
  --perturb \
  --feature_layer_name "8x8_block4" \
  --accumulation_steps 1 \
  --sigma 0.1 \
  --adv-eval-freq 1

python -m eval_scripts.eval-aa --data-dir 'data' \
  --log-dir 'checkpoints/202604/cifar10_0M/' \
  --desc 'PoolRepa0_WRN34-10Swish-8x8_cifar10_lr0p2_TRADES5_epoch200_bs512_ls0p1' \
  --eps 0.031372549

# CIFAR-10 50M
python -m train_scripts.train-wa-repa --data-dir 'data' \
  --log-dir 'checkpoints/cifar10_50M' \
  --desc 'PoolRepa0_vit-b-repa-pool_cifar10_lr1e-4_TRADES3_epoch2000_bs1024_ls0p4_lion_wd0p5' \
  --data cifar10s \
  --batch-size 1024 \
  --batch-size-validation 1024 \
  --model vit-b-repa-pool \
  --num-adv-epochs 2000 \
  --adv-eval-freq 25 \
  --lr 1e-4 \
  --weight-decay 0.5 \
  --beta 3.0 \
  --opt-type lion \
  --unsup-fraction 0.9 \
  --aux-data-filename 'data/50m.npz' \
  --ls 0.4 \
  --feat-reg-coeff 0.0 \
  --sigma 0.1 \
  --dual-time \
  --perturb \
  --feature_layer_name "8x8_block4" \
  --accumulation_steps 1

python -m eval_scripts.eval-last-aa --data-dir 'data' \
  --log-dir 'checkpoints/cifar10_50M' \
  --desc 'PoolRepa0_vit-b-repa-pool_cifar10_lr1e-4_TRADES3_epoch2000_bs1024_ls0p4_lion_wd0p5' \
  --eps 0.031372549

# nproc_per_node=NUM_GPUS
TIMM_FUSED_ATTN=2 OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 --master_port=$MASTER_PORT -m train_scripts.train-wa-repa-in1k --data-dir 'data' \
  --log-dir 'checkpoints/imagenet_ldit4M/202604/pre-vit-b-dinov3-baseaug' \
  --desc "PoolRepa12_pre-vit-b-dinov3_IN1k_lr1e-4_TRADES10_epoch100_bs1024_ls0p1_lion_wd0p0_clipgrad1p0" \
  --data imagenet \
  --augment base \
  --batch-size 1024 \
  --batch-size-validation 1024 \
  --model pre-vit-b-dinov3-repa-pool \
  --num-adv-epochs 100 \
  --opt-type lion \
  --lr 1e-4 \
  --weight-decay 0.0 \
  --beta 10.0 \
  --feat-reg-coeff 12.0 \
  --use-latent-repa \
  --lightning-dit-config LightningDiT/configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml \
  --lightning-dit-ckpt checkpoints/ldit_checkpoints/ldit-xl-img256-800ep.pt \
  --latent-dir data/imagenet_vae_latents_bf16/vavae_f16d32/imagenet_train_256 \
  --dit-feature-layers 14 \
  --dit-timestep 0.8 \
  --num-timesteps 2 \
  --mix-fixed-sampled-timesteps \
  --ls 0.1 \
  --accumulation_steps 1 \
  --attack-iter 3 \
  --attack-step "8/(255*3)" \
  --attack-eps 4/255 \
  --repa_dim 1152 \
  --clip-grad 1.0 \
  --unsup-fraction 0.8 \
  --aux-data-filename data/imagenet_ldit4M \
  --aux-latent-dir data/imagenet_vae_latents_bf16/vavae_f16d32/imagenet_ldit4M_256 \
  --adv-eval-freq 1

python -m eval_scripts.eval-last-aa --data-dir 'data' \
  --log-dir 'checkpoints/imagenet_ldit4M/202604/pre-vit-b-dinov3-baseaug' \
  --desc "PoolRepa12_pre-vit-b-dinov3_IN1k_lr1e-4_TRADES10_epoch100_bs1024_ls0p1_lion_wd0p0_clipgrad1p0" \
  --eps '0.0156862745'
