import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.data import get_data_info
from core.data import load_data

# from core.data import SEMISUP_DATASETS

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

# Distributed Training Setup
if "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])
else:
    args.local_rank = -1

args.distributed = args.local_rank != -1

if args.distributed:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")
    print(f"Initialized DDP on rank {args.local_rank}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce_tensor(tensor, n):
    """
    Average tensor across all processes.
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def aggregate_metrics(metrics_dict, device, distributed):
    """
    Aggregate metrics dictionary across processes.
    """
    if not distributed:
        return metrics_dict

    world_size = dist.get_world_size()
    aggregated = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            t = torch.tensor(v, device=device)
            t = reduce_tensor(t, world_size)
            aggregated[k] = t.item()
        else:
            aggregated[k] = v
    return aggregated


# Parse repa_class argument
if args.repa_class is not None:
    args.repa_class = [int(c.strip()) for c in args.repa_class.split(",")]
else:
    args.repa_class = []
# assert args.data in SEMISUP_DATASETS, f"Only data in {SEMISUP_DATASETS} is supported!"


DATA_DIR = os.path.join(args.data_dir, args.data)
LOG_DIR = os.path.join(args.log_dir, args.desc)
STATE_LAST = os.path.join(LOG_DIR, "state-last.pt")

# Which checkpoint to load: if logdir has state-last.pt always use it; else use resume_path if provided
checkpoint_to_load = None
if os.path.exists(STATE_LAST):
    checkpoint_to_load = STATE_LAST
elif args.resume_path:
    checkpoint_to_load = (
        args.resume_path if os.path.isfile(args.resume_path) else os.path.join(args.resume_path, "state-last.pt")
    )

# Ensure only rank 0 creates directories and writes logs
if args.local_rank in [-1, 0]:
    os.makedirs(LOG_DIR, exist_ok=True)
    WEIGHTS = os.path.join(LOG_DIR, "weights-best.pt")
else:
    WEIGHTS = ""  # Placeholder for other ranks

# Barrier to ensure directory is created by rank 0 before others proceed
if args.distributed:
    dist.barrier(device_ids=[args.local_rank])

info = get_data_info(DATA_DIR)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Only log on rank 0
if args.local_rank in [-1, 0]:
    logger = Logger(os.path.join(LOG_DIR, "log-train.log"), mode=("a" if checkpoint_to_load else "w"))
else:

    class DummyLogger:
        def log(self, *args, **kwargs):
            pass

    logger = DummyLogger()

if args.dummy_orth_feat:
    raise NotImplementedError("Dummy orthogonal feature extractor is not supported for ImageNet.")
elif args.use_disc_feat:
    raise NotImplementedError("Discriminative feature extractor is not supported for ImageNet.")
else:
    feature_model = None
    args.teacher_input = "image"

    if args.feat_reg_coeff and args.feat_reg_coeff > 0:
        teacher_type = args.teacher_type
        uses_latent_teacher = teacher_type == "lightning-dit" or args.use_latent_repa

        if teacher_type != "lightning-dit" and args.use_latent_repa:
            raise ValueError("--use-latent-repa is only valid with --teacher-type lightning-dit.")

        if args.cond_align and teacher_type != "lightning-dit":
            raise ValueError("--cond_align is only supported for the LightningDiT teacher.")

        if uses_latent_teacher:
            # LightningDiT feature extractor for ImageNet with pre-computed latents
            from core.models.lightning_dit.feature_extractor import LightningDiTFeatureExtractor

            if args.lightning_dit_config is None or args.lightning_dit_ckpt is None:
                raise ValueError("--lightning-dit-config and --lightning-dit-ckpt are required for latent-based REPA.")
            if args.latent_dir is None:
                raise ValueError("--latent-dir is required for latent-based REPA.")

            args.teacher_input = "latent"

            # Parse feature layers
            feature_layers = [int(x.strip()) for x in args.dit_feature_layers.split(",")]

            # Determine timestep: -1 means sampling, otherwise fixed
            timestep = None if args.dit_timestep < 0 else args.dit_timestep

            feature_model = LightningDiTFeatureExtractor(
                config_path=args.lightning_dit_config,
                checkpoint_path=args.lightning_dit_ckpt,
                feature_layers=feature_layers,
                timestep=timestep,
                timestep_low=args.dit_timestep_low,
                timestep_high=args.dit_timestep_high,
                device=device,
                freeze=True,
            )
            args.cond_embed_dim = feature_model.cond_embed_dim
            logger.log("Using LightningDiT feature extractor:")
            logger.log(f"  - Config: {args.lightning_dit_config}")
            logger.log(f"  - Checkpoint: {args.lightning_dit_ckpt}")
            logger.log(f"  - Feature layers: {feature_layers}")
            logger.log(
                f"  - Timestep: {f'sampled from ({args.dit_timestep_low}, {args.dit_timestep_high})' if timestep is None else timestep}"
            )
            logger.log(f"  - Num timesteps for averaging: {args.num_timesteps}")
        else:
            from core.models.ssl_teacher_feature_extractor import SSLTeacherFeatureExtractor

            args.teacher_input = "image"
            feature_model = SSLTeacherFeatureExtractor(
                teacher_type=teacher_type,
                model_name=args.teacher_model_name,
                pool=args.teacher_pool,
                device=device,
                freeze=True,
                standardize=getattr(args, "teacher_feat_standardize", False),
            )
            logger.log("Using self-supervised image teacher:")
            logger.log(f"  - Teacher type: {teacher_type}")
            logger.log(f"  - Teacher model: {feature_model.model_name}")
            logger.log(f"  - Teacher pool: {args.teacher_pool}")
            logger.log(f"  - Teacher feature standardization: {getattr(args, 'teacher_feat_standardize', False)}")

        if args.repa_dim != feature_model.hidden_size:
            raise ValueError(
                f"--repa_dim ({args.repa_dim}) must match teacher hidden size ({feature_model.hidden_size}) "
                f"for {args.teacher_type} alignment."
            )
    else:
        logger.log(f"Not using repa because feat_reg_coeff is {args.feat_reg_coeff}.")


if args.local_rank in [-1, 0]:
    with open(os.path.join(LOG_DIR, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

if args.repa_class:
    logger.log(f"REPA loss will only be applied to classes: {args.repa_class}")
else:
    logger.log("REPA loss will be applied to all classes.")


BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
# divide batch size by the number of processes
if args.distributed and dist.get_world_size() > 1:
    BATCH_SIZE //= dist.get_world_size()
    BATCH_SIZE_VALIDATION //= dist.get_world_size()
NUM_ADV_EPOCHS = args.num_adv_epochs
if args.stop_epoch is not None and args.stop_epoch < 1:
    raise ValueError("--stop-epoch must be >= 1.")
if args.accumulation_steps > 1:
    if args.local_rank in [-1, 0]:
        logger.log("Using gradient accumulation.")
    assert BATCH_SIZE % args.accumulation_steps == 0, "Batch size must be divisible by accumulation steps."
    BATCH_SIZE //= args.accumulation_steps

# logger.log("Using device: {}".format(device))
if args.debug:
    NUM_ADV_EPOCHS = 1
STOP_EPOCH = NUM_ADV_EPOCHS if args.stop_epoch is None else min(args.stop_epoch, NUM_ADV_EPOCHS)

# To speed up training
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True


# Load data

seed(args.seed)

# if args.use_latent_repa and args.feat_reg_coeff and args.feat_reg_coeff > 0:
if getattr(args, "teacher_input", "image") == "latent":
    # Use PairedImageLatentDataset for latent-based REPA with semi-supervised sampling
    from core.data import load_paired_data

    train_dataset, test_dataset, train_dataloader, test_dataloader = load_paired_data(
        data_dir=DATA_DIR,
        latent_dir=args.latent_dir,
        batch_size=BATCH_SIZE,
        batch_size_test=BATCH_SIZE_VALIDATION,
        num_workers=4,
        use_augmentation=args.augment,
        distributed=args.distributed,
        aux_data_filename=args.aux_data_filename,
        aux_latent_dir=args.aux_latent_dir,
        unsup_fraction=args.unsup_fraction,
    )

    logger.log(f"Using PairedImageLatentDataset for REPA training:")
    logger.log(f"  - Data dir: {DATA_DIR}")
    logger.log(f"  - Latent dir: {args.latent_dir}")
    if args.aux_data_filename:
        logger.log(f"  - Aux image dir: {args.aux_data_filename}")
        logger.log(f"  - Aux latent dir: {args.aux_latent_dir}")
        logger.log(f"  - Unsup fraction: {args.unsup_fraction}")
    logger.log(f"  - Train samples: {len(train_dataset)}")
    logger.log(f"  - Test samples: {len(test_dataset)}")
else:
    # Standard data loading
    train_dataset, test_dataset, train_dataloader, test_dataloader = load_data(
        DATA_DIR,
        BATCH_SIZE,
        BATCH_SIZE_VALIDATION,
        use_augmentation=args.augment,
        use_consistency=args.consistency,
        shuffle_train=True,
        aux_data_filename=args.aux_data_filename,
        unsup_fraction=args.unsup_fraction,
        validation=False,
        num_workers=4,
        distributed=args.distributed,
    )

del train_dataset, test_dataset

# get my diffusion feature extractor and pass it to the trainer
if args.sigma and args.sigma < 0:
    raise ValueError("Sigma must be None (using sampling) or non-negative.")


# Adversarial Training

seed(args.seed)
if args.tau:
    if args.local_rank in [-1, 0]:
        logger.log("Using WA.")
    trainer = WATrainerRepa(info, args, feature_model, device=device, DDP=args.distributed)
else:
    raise NotImplementedError("Only Weight Averaging is supported for now.")
    trainer = Trainer(info, args, device=device, DDP=args.distributed)
last_lr = args.lr
res = None
old_score = [0.0, 0.0]
test_adv_acc = 0.0


if NUM_ADV_EPOCHS > 0:
    logger.log("\n\n")
    if checkpoint_to_load and os.path.exists(os.path.join(LOG_DIR, "stats_adv.csv")) and args.local_rank in [-1, 0]:
        metrics = pd.read_csv(os.path.join(LOG_DIR, "stats_adv.csv"))
    else:
        metrics = pd.DataFrame()

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        acc = trainer.eval(test_dataloader)
        if args.distributed:
            acc = aggregate_metrics({"acc": acc}, device, args.distributed)["acc"]
        logger.log("Standard Accuracy-\tTest: {:2f}%.".format(acc * 100))

    old_score = [acc, 0.0]
    if args.stop_epoch is None or STOP_EPOCH == NUM_ADV_EPOCHS:
        logger.log("RST Adversarial training for {} epochs".format(NUM_ADV_EPOCHS))
    else:
        logger.log(
            "RST Adversarial training for {} epochs (stopping early after epoch {})".format(NUM_ADV_EPOCHS, STOP_EPOCH)
        )
    trainer.init_optimizer(args.num_adv_epochs)

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

if NUM_ADV_EPOCHS > 0 and start_epoch > STOP_EPOCH:
    logger.log(
        f"Current start epoch {start_epoch} is already past stop epoch {STOP_EPOCH}; no additional training will run."
    )

for epoch in range(start_epoch, STOP_EPOCH + 1):
    # Set epoch for DistributedSampler
    if args.distributed:
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        elif hasattr(train_dataloader.batch_sampler, "set_epoch"):
            train_dataloader.batch_sampler.set_epoch(epoch)

    start = time.time()
    logger.log("======= Epoch {} =======".format(epoch))

    if args.scheduler:
        last_lr = trainer.scheduler.get_last_lr()[0]

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        res = trainer.train(
            train_dataloader,
            epoch=epoch,
            adversarial=True,
            verbose=(args.local_rank in [-1, 0]),
            accumulation_steps=args.accumulation_steps,
        )
    if args.distributed:
        res = aggregate_metrics(res, device, args.distributed)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        test_acc = trainer.eval(test_dataloader)
    if args.distributed:
        test_acc = aggregate_metrics({"acc": test_acc}, device, args.distributed)["acc"]

    logger.log("Loss: {:.4f}.\tLR: {:.8f}".format(res["loss"], last_lr))
    if "clean_acc" in res:
        logger.log("Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.".format(res["clean_acc"] * 100, test_acc * 100))
    else:
        logger.log("Standard Accuracy-\tTest: {:.2f}%.".format(test_acc * 100))

    # Only save metrics on rank 0
    if args.local_rank in [-1, 0]:
        epoch_metrics = {"train_" + k: v for k, v in res.items()}
        epoch_metrics.update(
            {
                "epoch": epoch,
                "lr": last_lr,
                "test_clean_acc": test_acc,
                "test_adversarial_acc": "",
            }
        )

    if epoch % args.adv_eval_freq == 0 or epoch == NUM_ADV_EPOCHS or epoch == STOP_EPOCH:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
        if args.distributed:
            test_adv_acc = aggregate_metrics({"acc": test_adv_acc}, device, args.distributed)["acc"]

        logger.log(
            "Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.".format(
                res["adversarial_acc"] * 100, test_adv_acc * 100
            )
        )
        if args.local_rank in [-1, 0]:
            epoch_metrics.update({"test_adversarial_acc": test_adv_acc})
            if test_adv_acc >= old_score[1]:
                old_score[0], old_score[1] = test_acc, test_adv_acc
                trainer.save_model(WEIGHTS)
    else:
        logger.log("Adversarial Accuracy-\tTrain: {:.2f}%.".format(res["adversarial_acc"] * 100))

    if args.local_rank in [-1, 0]:
        if epoch % 10 == 0 or epoch == STOP_EPOCH:
            trainer.save_model_resume(os.path.join(LOG_DIR, "state-last.pt"), epoch)
        if epoch % 400 == 0:
            shutil.copyfile(WEIGHTS, os.path.join(LOG_DIR, f"weights-best-epoch{str(epoch)}.pt"))
        if epoch % 50 == 0:
            trainer.save_model_resume(os.path.join(LOG_DIR, f"state-epoch{str(epoch)}.pt"), epoch)
        elif epoch % 90 == 0:
            trainer.save_model_resume(os.path.join(LOG_DIR, f"state-epoch{str(epoch)}.pt"), epoch)
        elif epoch == (STOP_EPOCH - 10):
            trainer.save_model_resume(os.path.join(LOG_DIR, f"state-epoch{str(epoch)}.pt"), epoch)

        logger.log("Time taken: {}".format(format_time(time.time() - start)))
        # metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
        metrics = pd.concat([metrics, pd.DataFrame(epoch_metrics, index=[0])], ignore_index=True)
        metrics.to_csv(os.path.join(LOG_DIR, "stats_adv.csv"), index=False)

if args.stop_epoch is not None and 0 < STOP_EPOCH < NUM_ADV_EPOCHS and start_epoch <= STOP_EPOCH:
    logger.log(f"Reached requested stop epoch {STOP_EPOCH}; exiting early.")


# Record metrics

train_acc = res["clean_acc"] if res is not None and "clean_acc" in res else trainer.eval(train_dataloader)
if args.distributed and (res is None or "clean_acc" not in res):
    train_acc = aggregate_metrics({"acc": train_acc}, device, args.distributed)["acc"]

logger.log("\nTraining completed.")
logger.log("Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.".format(train_acc * 100, old_score[0] * 100))
if NUM_ADV_EPOCHS > 0 and res is not None:
    logger.log(
        "Adversarial Accuracy-\tTrain: {:.2f}%.\tEval: {:.2f}%.".format(
            res["adversarial_acc"] * 100, old_score[1] * 100
        )
    )
elif NUM_ADV_EPOCHS > 0:
    logger.log("Adversarial Accuracy-\tEval: {:.2f}%.".format(old_score[1] * 100))

logger.log("Script Completed.")

