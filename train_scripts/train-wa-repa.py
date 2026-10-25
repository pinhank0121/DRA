import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data
from core.data import SEMISUP_DATASETS

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train
from core.utils import Trainer
from core.utils import seed

from gowal21uncovering.utils import WATrainerRepa

import sys

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "edm"))
sys.path.append(os.path.join(cwd, "edm", "dnnlib"))

from edm.get_diffusion import get_feature_model
from edm.get_dummy_orth import get_dummy_feature_model
from core.models import create_model
from core.models import DiscFeatExtractorWrapper


# Setup

parse = parser_train()
args = parse.parse_args()

# Parse repa_class argument
if args.repa_class is not None:
    args.repa_class = [int(c.strip()) for c in args.repa_class.split(',')]
else:
    args.repa_class = []
assert args.data in SEMISUP_DATASETS, f"Only data in {SEMISUP_DATASETS} is supported!"


DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
os.makedirs(LOG_DIR, exist_ok=True)
WEIGHTS = os.path.join(LOG_DIR, "weights-best.pt")
STATE_LAST = os.path.join(LOG_DIR, "state-last.pt")

# Which checkpoint to load: if logdir has state-last.pt always use it; else use resume_path if provided
checkpoint_to_load = None
if os.path.exists(STATE_LAST):
    checkpoint_to_load = STATE_LAST
elif args.resume_path:
    checkpoint_to_load = args.resume_path if os.path.isfile(args.resume_path) else os.path.join(args.resume_path, "state-last.pt")

info = get_data_info(DATA_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = Logger(os.path.join(LOG_DIR, "log-train.log"), mode=('a' if checkpoint_to_load else 'w'))

if args.dummy_orth_feat:
    logger.log("Using dummy orthogonal feature extractor.")
    print("Using dummy orthogonal feature extractor.")
    num_classes = 10 if args.data == "cifar10s" else None
    num_classes = 100 if args.data == "cifar100s" else num_classes
    feature_model = get_dummy_feature_model(
        # num_classes=10 if args.data == "cifar10" else None,
        num_classes=num_classes,
        feature_dims=(args.repa_dim, 1, 1),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    args.cond_embed_dim = feature_model.num_classes
elif args.use_disc_feat:
    print(f"Using discriminative feature extractor: {args.disc_feat_model}")
    feature_model = DiscFeatExtractorWrapper(
        model = create_model(args.disc_feat_model, args.normalize, info, device, repa_dim=256),
        path=args.disc_feat_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    args.cond_embed_dim = 0
elif getattr(args, "cached_feature_teacher", False):
    # Off-the-shelf diffusion teacher (App. J): alignment targets are cached pooled
    # feature vectors (e.g. SD-1.5 mid-block, precompute-cifar-sd-features.py). The
    # "teacher" is an identity adapter into the latent-REPA plumbing.
    if args.latent_dir is None:
        raise ValueError("--latent-dir (cached feature dir) is required for --cached-feature-teacher.")

    class _CachedFeatureTeacher(torch.nn.Module):
        def forward(self, feats, labels=None, **kwargs):
            return {"features": [feats.unsqueeze(1)]}

    feature_model = _CachedFeatureTeacher()
    args.teacher_input = "latent"
    args.cond_embed_dim = 0
    logger.log(f"Using cached pooled feature targets from {args.latent_dir} (repa_dim={args.repa_dim})")
elif args.use_latent_repa:
    # Off-the-shelf diffusion teacher (cf. App. J): ImageNet LightningDiT as the
    # alignment teacher for a CIFAR student, consuming pre-computed VA-VAE latents of the
    # 32->256 upsampled training images (see train_scripts/precompute-cifar-vae-latents.py).
    # The teacher never saw CIFAR; labels are forced to the null class (unconditional) since
    # CIFAR labels are meaningless in the teacher's ImageNet class space.
    from core.models.lightning_dit.feature_extractor import LightningDiTFeatureExtractor

    if args.lightning_dit_config is None or args.lightning_dit_ckpt is None:
        raise ValueError("--lightning-dit-config and --lightning-dit-ckpt are required for latent-based REPA.")
    if args.latent_dir is None:
        raise ValueError("--latent-dir (CIFAR latent cache) is required for latent-based REPA.")

    class _UncondTeacher(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.timestep = inner.timestep
            self.timestep_low = inner.timestep_low
            self.timestep_high = inner.timestep_high

        def forward(self, latents, labels=None, **kwargs):
            return self.inner(latents, labels=None, **kwargs)

    _feature_layers = [int(x.strip()) for x in args.dit_feature_layers.split(",")]
    _timestep = None if args.dit_timestep < 0 else args.dit_timestep
    feature_model = _UncondTeacher(LightningDiTFeatureExtractor(
        config_path=args.lightning_dit_config,
        checkpoint_path=args.lightning_dit_ckpt,
        feature_layers=_feature_layers,
        timestep=_timestep,
        timestep_low=args.dit_timestep_low,
        timestep_high=args.dit_timestep_high,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        freeze=True,
    ))
    args.teacher_input = "latent"
    args.cond_embed_dim = 0
    logger.log("Using off-the-shelf LightningDiT teacher (unconditional) with cached CIFAR latents:")
    logger.log(f"  - Checkpoint: {args.lightning_dit_ckpt}")
    logger.log(f"  - Latent cache: {args.latent_dir}")
    logger.log(f"  - Timestep: {_timestep} (mix sampled: {args.mix_fixed_sampled_timesteps}, n={args.num_timesteps})")
else:
    feature_model = get_feature_model(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        sigma=args.sigma,
        feature_layer_name=args.feature_layer_name,
        unconditional=args.unconditional,
        perturb=args.perturb,
        cfg=args.CFG,
        dataset=args.data,
        dual_time=args.dual_time,
        duplicate_time=args.duplicate_time,
    )
    args.cond_embed_dim = feature_model.cond_embed_dim


with open(os.path.join(LOG_DIR, "args.txt"), "w") as f:
    json.dump(args.__dict__, f, indent=4)

if args.repa_class:
    logger.log(f"REPA loss will only be applied to classes: {args.repa_class}")
else:
    logger.log("REPA loss will be applied to all classes.")



BATCH_SIZE = args.batch_size
if args.accumulation_steps > 1:
    print("Using gradient accumulation.")
    assert BATCH_SIZE % args.accumulation_steps == 0, "Batch size must be divisible by accumulation steps."
    BATCH_SIZE //= args.accumulation_steps
BATCH_SIZE_VALIDATION = args.batch_size_validation
NUM_ADV_EPOCHS = args.num_adv_epochs

logger.log("Using device: {}".format(device))
if args.debug:
    NUM_ADV_EPOCHS = 1

# To speed up training
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False


# Load data

seed(args.seed)
train_dataset, test_dataset, eval_dataset, train_dataloader, test_dataloader, eval_dataloader = load_data(
    DATA_DIR,
    BATCH_SIZE,
    BATCH_SIZE_VALIDATION,
    use_augmentation=args.augment,
    use_consistency=args.consistency,
    shuffle_train=True,
    aux_data_filename=args.aux_data_filename,
    unsup_fraction=args.unsup_fraction,
    validation=True,
    num_workers=4,
)
if args.use_latent_repa or getattr(args, "cached_feature_teacher", False):
    # Dataset then yields (image, latent, label); the trainer routes latents to the teacher.
    train_dataset.attach_latent_cache(args.latent_dir)
    logger.log(f"Attached latent cache ({len(train_dataset)} samples) from {args.latent_dir}")
del train_dataset, test_dataset, eval_dataset

# get my diffusion feature extractor and pass it to the trainer
if args.sigma and args.sigma < 0:
    raise ValueError("Sigma must be None (using sampling) or non-negative.")




# Adversarial Training

seed(args.seed)
if args.tau:
    print("Using WA.")
    trainer = WATrainerRepa(info, args, feature_model)
else:
    raise NotImplementedError("Only Weight Averaging is supported for now.")
    trainer = Trainer(info, args)
last_lr = args.lr


if NUM_ADV_EPOCHS > 0:
    logger.log("\n\n")
    if checkpoint_to_load and os.path.exists(os.path.join(LOG_DIR, "stats_adv.csv")):
        metrics = pd.read_csv(os.path.join(LOG_DIR, "stats_adv.csv"))
    else:
        metrics = pd.DataFrame()
    with torch.no_grad():
        logger.log("Standard Accuracy-\tTest: {:2f}%.".format(trainer.eval(test_dataloader) * 100))

    old_score = [0.0, 0.0]
    logger.log("RST Adversarial training for {} epochs".format(NUM_ADV_EPOCHS))
    trainer.init_optimizer(args.num_adv_epochs)
    test_adv_acc = 0.0

if checkpoint_to_load:
    if os.path.exists(checkpoint_to_load):
        logger.log(f"Using checkpoint: {checkpoint_to_load}")
        start_epoch = trainer.load_model_resume(checkpoint_to_load) + 1
        logger.log(f"Resuming at epoch {start_epoch}")
    else:
        logger.log(f"Checkpoint not found at {checkpoint_to_load}; starting from epoch 1.")
        start_epoch = 1
else:
    start_epoch = 1

for epoch in range(start_epoch, NUM_ADV_EPOCHS + 1):
    start = time.time()
    logger.log("======= Epoch {} =======".format(epoch))

    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]

    res = trainer.train(
        train_dataloader, epoch=epoch, adversarial=True, verbose=True, accumulation_steps=args.accumulation_steps
    )
    test_acc = trainer.eval(test_dataloader)

    logger.log("Loss: {:.4f}.\tLR: {:.4f}".format(res["loss"], last_lr))
    if "clean_acc" in res:
        logger.log("Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.".format(res["clean_acc"] * 100, test_acc * 100))
    else:
        logger.log("Standard Accuracy-\tTest: {:.2f}%.".format(test_acc * 100))
    epoch_metrics = {"train_" + k: v for k, v in res.items()}
    epoch_metrics.update(
        {
            "epoch": epoch,
            "lr": last_lr,
            "test_clean_acc": test_acc,
            "test_adversarial_acc": "",
            "eval_adversarial_acc": "",
        }
    )

    if epoch % args.adv_eval_freq == 0 or epoch == NUM_ADV_EPOCHS:
        test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
        logger.log(
            "Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.".format(
                res["adversarial_acc"] * 100, test_adv_acc * 100
            )
        )
        epoch_metrics.update({"test_adversarial_acc": test_adv_acc})
    else:
        logger.log("Adversarial Accuracy-\tTrain: {:.2f}%.".format(res["adversarial_acc"] * 100))
    if epoch % args.adv_eval_freq == 0 or epoch == NUM_ADV_EPOCHS:
        eval_adv_acc = trainer.eval(eval_dataloader, adversarial=True)
        logger.log("Adversarial Accuracy-\tEval: {:.2f}%.".format(eval_adv_acc * 100))
        epoch_metrics["eval_adversarial_acc"] = eval_adv_acc

        if eval_adv_acc >= old_score[1]:
            old_score[0], old_score[1] = test_acc, eval_adv_acc
            trainer.save_model(WEIGHTS)
    # trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))
    if epoch % 10 == 0:
        trainer.save_model_resume(os.path.join(LOG_DIR, "state-last.pt"), epoch)
    if epoch % 400 == 0:
        # shutil.copyfile(WEIGHTS, os.path.join(LOG_DIR, f"weights-best-epoch{str(epoch)}.pt"))
        trainer.save_model_resume(os.path.join(LOG_DIR, f"state-epoch{str(epoch)}.pt"), epoch)
    elif epoch == (NUM_ADV_EPOCHS - 100):
        trainer.save_model_resume(os.path.join(LOG_DIR, f"state-epoch{str(epoch)}.pt"), epoch)
    elif epoch == (NUM_ADV_EPOCHS - 50):
        trainer.save_model_resume(os.path.join(LOG_DIR, f"state-epoch{str(epoch)}.pt"), epoch)

    logger.log("Time taken: {}".format(format_time(time.time() - start)))
    # metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
    metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
    metrics.to_csv(os.path.join(LOG_DIR, "stats_adv.csv"), index=False)


# Record metrics

train_acc = res["clean_acc"] if "clean_acc" in res else trainer.eval(train_dataloader)
logger.log("\nTraining completed.")
logger.log("Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.".format(train_acc * 100, old_score[0] * 100))
if NUM_ADV_EPOCHS > 0:
    logger.log(
        "Adversarial Accuracy-\tTrain: {:.2f}%.\tEval: {:.2f}%.".format(
            res["adversarial_acc"] * 100, old_score[1] * 100
        )
    )

logger.log("Script Completed.")
