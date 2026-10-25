import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import copy
import json
import sys
import torch
import torch.nn as nn
from timm.optim import Lion

# Increase recursion limit to handle deep model structures
sys.setrecursionlimit(10000)
import torch.nn.functional as F

from core.attacks import create_attack
from core.attacks import CWLoss
from core.metrics import accuracy
from core.models import create_model

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Trainer
from core.utils import set_bn_momentum
from core.utils import seed

from .trades import (
    trades_loss,
    trades_loss_LSE,
    trades_loss_repa,
    trades_loss_repa_teacher,
    standard_loss_repa,
    repa_distillation_loss,
    trades_supervision,
    standard_loss_supervision,
)
from .cutmix import cutmix

def get_num_samples(data):
    if "imagenet" in data:
        return 1281167
    elif "cifar" in data:
        return 50000
    elif "tiny-imagenet" in data:
        return 100000
    else:
        raise ValueError(f"Invalid dataset: {data}")


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WATrainerRepa(Trainer):
    """
    Helper class for training a deep neural network with model weight averaging (identical to Gowal et al, 2020).
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """

    def __init__(self, info, args, feature_model=None, device=None, DDP=False):
        super(WATrainerRepa, self).__init__(info, args, device=device, DDP=DDP)

        seed(args.seed)
        # Avoid deep copy recursion issues by using state_dict
        # WA model should NOT be wrapped in DDP/DataParallel as we just update its weights manually
        self.DDP = DDP
        use_repa = args.feat_reg_coeff is not None and args.feat_reg_coeff > 0.0
        self.wa_model = create_model(args.model, args.normalize, info, self.device, args.repa_dim, DataParallel=not DDP, use_repa=use_repa)
        
        # Copy weights from current model (handling DDP wrap if present)
        if self.DDP:
            model_to_copy = self.model.module if hasattr(self.model, 'module') else self.model
        else:
            model_to_copy = self.model
        self.wa_model.load_state_dict(model_to_copy.state_dict())
        
        self.eval_attack = create_attack(
            self.wa_model, CWLoss, args.attack, args.attack_eps, 4 * args.attack_iter, args.attack_step
        )

        # num_samples = 1281167 if "imagenet" in self.params.data else 50000
        # num_samples = 50000 if "cifar" in self.params.data else 73257
        # num_samples = 100000 if "tiny-imagenet" in self.params.data else num_samples
        if self.params.data in ["cifar10", "cifar10s", "svhn", "svhns"]:
            self.num_classes = 10
        elif self.params.data in ["cifar100", "cifar100s"]:
            self.num_classes = 100
        elif self.params.data in ["cifar100-20"]:
            self.num_classes = 20
        elif self.params.data == "tiny-imagenet":
            self.num_classes = 200
        elif self.params.data == "cifar10-binary":
            self.num_classes = 2
        elif self.params.data == "imagenet":
            self.num_classes = 1000
        else:
            raise ValueError(f"Invalid dataset: {self.params.data}")
        self.update_steps = int(np.floor(get_num_samples(self.params.data) / self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps
        self.feature_model = feature_model
        # self.feature_model = torch.nn.DataParallel(feature_model)
        # assert self.params.feat_reg_coeff is not None, "Feature regularization coefficient must be provided."
        if self.params.feat_reg_coeff is None:
            print("Warning: Not using feat_reg_coeff. REPA disabled.")
        if self.params.feat_reg_adv:
            print("Input for diffusion uses adv examples instead of natural examples.")
        if self.params.alignment_v3:
            print("Also attack representation alignment for adversarial example augmentation.")
        if self.params.CFG:
            print("Using CFG for representation alignment.")
        print("Feature regularization coefficient: {}".format(self.params.feat_reg_coeff))
        if hasattr(self.params, "sup_enable") and self.params.sup_enable:
            print("Using auxiliary supervision head training.")
            print("Auxiliary supervision weight: {}".format(self.params.sup_weight))

    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and schedulers.
        """

        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if "batchnorm" in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=0.0)]
            return groups

        if self.params.opt_type == "adam":
            self.optimizer = torch.optim.Adam(
                group_weight(self.model),
                lr=self.params.lr,
                weight_decay=self.params.weight_decay,
            )
        elif self.params.opt_type == "lion":
            self.optimizer = Lion(
                group_weight(self.model),
                lr=self.params.lr,
                weight_decay=self.params.weight_decay,
            )
        elif self.params.opt_type == "muon":
            # Muon for 2D+ weight matrices, AdamW fallback for 1D params (timm routes
            # by ndim); embeddings/cls/head stay on the AdamW fallback per standard
            # Muon practice. adjust_lr_fn='match_rms_adamw' (default) -> lr is
            # AdamW-scale for both groups.
            from timm.optim import Muon

            _fallback_keys = ("pos_embed", "cls_token", "patch_embed", "head.")
            _muon_params, _fallback_params = [], []
            for n, p in self.model.named_parameters():
                if any(k in n for k in _fallback_keys):
                    _fallback_params.append(p)
                else:
                    _muon_params.append(p)
            self.optimizer = Muon(
                [
                    dict(params=_muon_params),
                    dict(params=_fallback_params, use_fallback=True),
                ],
                lr=self.params.lr,
                weight_decay=self.params.weight_decay,
                momentum=0.95,
                nesterov=True,
            )
        else:
            self.optimizer = torch.optim.SGD(
                group_weight(self.model),
                lr=self.params.lr,
                weight_decay=self.params.weight_decay,
                momentum=0.9,
                nesterov=self.params.nesterov,
            )
        # print number of trainbale parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Number of trainable parameters: {}".format(num_params))
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)

    def train(self, dataloader, epoch=0, adversarial=False, verbose=True, accumulation_steps=1, repa_st_abl=False):
        """
        Run one epoch of training with gradient accumulation support.

        Args:
            dataloader: Data loader for training
            epoch: Current epoch number
            adversarial: Whether to use adversarial training
            verbose: Whether to display progress bar
            accumulation_steps: Number of batches to accumulate gradients over before updating weights
            repa_abl: Whether to enable REPA ablation (Standard Training + REPA)
        """
        metrics = pd.DataFrame()
        self.model.train()
        if repa_st_abl:
            assert not adversarial

        update_iter = 0

        # Optional linear warmup of the alignment coefficient over the first N epochs.
        eff_feat_reg_coeff = self.params.feat_reg_coeff
        warmup_epochs = getattr(self.params, "feat_reg_warmup_epochs", 0) or 0
        if eff_feat_reg_coeff and warmup_epochs > 0:
            eff_feat_reg_coeff = eff_feat_reg_coeff * min(1.0, epoch / warmup_epochs)

        log_grad_cosine = getattr(self.params, "log_grad_cosine", False)
        grad_cosine_interval = max(1, getattr(self.params, "grad_cosine_interval", 20) or 20)

        # Handle verbose only on rank 0
        iterator = enumerate(dataloader)
        if verbose:
            iterator = enumerate(tqdm(dataloader, desc="Epoch {}: ".format(epoch)))

        for batch_idx, data in iterator:
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 0:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 1:
                set_bn_momentum(self.model, momentum=0.01)

            # Zero gradients only at the start of accumulation cycle
            if batch_idx % accumulation_steps == 0:
                self.optimizer.zero_grad()

            # Support datasets that return:
            # - (x, y): standard
            # - (x, y_binary, y_orig10): supervision auxiliary head
            # - (x, latent, y): latent-based REPA (ImageNet with LightningDiT)
            latent = None
            y_orig = None
            if isinstance(data, (list, tuple)) and len(data) == 3:
                if getattr(self.params, "teacher_input", None) == "latent":
                    # Latent-based ImageNet REPA mode: (image, latent, label)
                    x, latent, y = data
                else:
                    # Supervision mode: (image, y_binary, y_orig10)
                    x, y, y_orig = data
            else:
                x, y = data
            if self.params.consistency:
                x_aug1, x_aug2, y = x[0].to(self.device), x[1].to(self.device), y.to(self.device)
                if self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss_consistency(x_aug1, x_aug2, y, beta=self.params.beta)

            else:
                if self.params.CutMix and not (y_orig is not None and self.params.sup_enable) and latent is None:
                    x_all, y_all = torch.tensor([]), torch.tensor([])
                    for i in range(4):  # 128 x 4 = 512 or 256 x 4 = 1024
                        x_tmp, y_tmp = x.detach(), y.detach()
                        x_tmp, y_tmp = cutmix(x_tmp, y_tmp, alpha=1.0, beta=1.0, num_classes=self.num_classes)
                        x_all = torch.cat((x_all, x_tmp), dim=0)
                        y_all = torch.cat((y_all, y_tmp), dim=0)
                    x, y = x_all.to(self.device), y_all.to(self.device)
                else:
                    x, y = x.to(self.device), y.to(self.device)
                    if latent is not None:
                        latent = latent.to(self.device)

                # Supervision auxiliary head path (for CIFAR10-binary with original labels)
                if y_orig is not None and self.params.sup_enable:
                    y_orig = y_orig.to(self.device)
                    if adversarial:
                        loss, batch_metrics = self.trades_loss_supervision(x, y, y_orig, beta=self.params.beta)
                    else:
                        loss, batch_metrics = self.standard_loss_supervision(x, y, y_orig)
                elif adversarial:
                    if self.params.beta is not None and self.params.mart:
                        raise NotImplementedError("MART loss not implemented for adversarial training.")
                        loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                    elif self.params.beta is not None and self.params.LSE:
                        raise NotImplementedError("LSE loss not implemented for adversarial training.")
                        loss, batch_metrics = self.trades_loss_LSE(x, y, beta=self.params.beta)
                    elif self.params.feat_reg_coeff is not None and self.params.beta is not None:
                        if latent is not None:
                            # ImageNet REPA with a latent teacher input.
                            loss, batch_metrics = self._trades_loss_repa_teacher(
                                x, latent, y,
                                feat_reg_coeff=eff_feat_reg_coeff,
                                log_grad_cosine=log_grad_cosine and (batch_idx % grad_cosine_interval == 0),
                            )
                        elif self.params.alignment_v3:
                            raise NotImplementedError("not supporting gradient accumulation")
                            assert not self.params.feat_reg_adv, "Alignmentv3 and feat_reg_adv cannot be used together."
                            loss, batch_metrics = self.trades_loss_repa_v3(
                                x, y, beta=self.params.beta, feat_reg_coeff=self.params.feat_reg_coeff
                            )
                        elif self.params.feat_reg_adv:
                            raise NotImplementedError("not supporting gradient accumulation")
                            # refer to alignment input: Both diff/cls uses adv example atked from cls
                            loss, batch_metrics = self.trades_loss_repa_v2(
                                x, y, beta=self.params.beta, feat_reg_coeff=self.params.feat_reg_coeff
                            )
                        else:
                            if self.params.data == "imagenet":
                                # ImageNet REPA with an image teacher input.
                                loss, batch_metrics = self._trades_loss_repa_teacher(
                                    x,
                                    x,
                                    y,
                                    feat_reg_coeff=eff_feat_reg_coeff,
                                    log_grad_cosine=log_grad_cosine and (batch_idx % grad_cosine_interval == 0),
                                )
                            else:
                                # diff uses natural image and cls uses adv example
                                loss, batch_metrics = self.trades_loss_repa(
                                    x,
                                    y,
                                    feat_reg_coeff=eff_feat_reg_coeff,
                                    CFG=self.params.CFG,
                                )
                    elif self.params.beta is not None:
                        # raise NotImplementedError("TRADES loss not implemented for adversarial training.")
                        loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                    else:
                        loss, batch_metrics = self.adversarial_loss(x, y)
                elif repa_st_abl:
                    loss, batch_metrics = self.standard_loss_repa(x, y, feat_reg_coeff=self.params.feat_reg_coeff)
                else:
                    loss, batch_metrics = self.standard_loss(x, y)

            # Scale the loss according to accumulation steps
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            # Only update weights after accumulating gradients for specified number of steps
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                if self.params.clip_grad:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
                self.optimizer.step()
                if self.params.scheduler in ["cyclic"]:
                    self.scheduler.step()
                update_iter += 1

                global_step = (epoch - 1) * self.update_steps + update_iter
                
                # Handle EMA update - wa_model is local, model might be DDP
                model_to_avg = self.model.module if hasattr(self.model, 'module') else self.model
                ema_update(
                    self.wa_model,
                    model_to_avg,
                    global_step,
                    decay_rate=self.params.tau,
                    warmup_steps=self.warmup_steps,
                    dynamic_decay=True,
                )

            # metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
            metrics = pd.concat([metrics, pd.DataFrame(batch_metrics, index=[0])], ignore_index=True)

        if self.params.scheduler in ["step", "converge", "cosine", "cosinew"]:
            self.scheduler.step()
        
        # Update BN using local model (shadow) which is self.wa_model, and source model
        # If distributed, we might want to sync BN statistics across processes, but standard BN 
        # update usually does forward passes to accumulate stats.
        # `update_bn` implementation iterates over modules and copies running_mean/var.
        # Since wa_model is updated via EMA from the DDP model (which syncs grads), 
        # the running statistics of the DDP model (if using SyncBN) should be somewhat synced,
        # but normally BN stats track the activation statistics. 
        # The `update_bn` function below copies stats from `self.model` to `self.wa_model`.
        # If `self.model` is DDP, its buffers are synced if SyncBN is used, or local otherwise.
        # Here we assume standard behavior:
        if self.DDP:
            model_to_copy = self.model.module if hasattr(self.model, 'module') else self.model
        else:
            model_to_copy = self.model
        update_bn(self.wa_model, model_to_copy)
        
        return dict(metrics.mean())

    def standard_loss_supervision(self, x, y_binary, y_orig10):
        loss, batch_metrics = standard_loss_supervision(
            self.model,
            x,
            y_binary,
            y_orig10,
            sup_weight=self.params.sup_weight,
        )
        return loss, batch_metrics

    def trades_loss_supervision(self, x, y_binary, y_orig10, beta):
        loss, batch_metrics = trades_supervision(
            self.model,
            x,
            y_binary,
            y_orig10,
            self.optimizer,
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            beta=beta if beta is not None else 1.0,
            attack=self.params.attack,
            label_smoothing=self.params.ls,
            sup_weight=self.params.sup_weight,
        )
        return loss, batch_metrics

    def standard_loss_repa(self, x, y, feat_reg_coeff):
        """
        TRADES training.
        """
        loss, batch_metrics = standard_loss_repa(
            self.model,
            x,
            y,
            self.feature_model,
            feat_reg_coeff=feat_reg_coeff,
            cond_align=self.params.cond_align,
        )
        return loss, batch_metrics

    def trades_loss_repa(self, x, y, feat_reg_coeff, CFG=False):
        """
        TRADES training with image-based REPA (for CIFAR with EDM).
        """
        repa_class = getattr(self.params, 'repa_class', [])
        loss, batch_metrics = trades_loss_repa(
            self.model,
            x,
            y,
            self.optimizer,
            self.feature_model,
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            attack=self.params.attack,
            label_smoothing=self.params.ls,
            use_cutmix=self.params.CutMix,
            feat_reg_coeff=feat_reg_coeff,
            beta=self.params.beta,
            CFG=CFG,
            cond_align=self.params.cond_align,
            repa_class=repa_class,
        )
        return loss, batch_metrics

    def _trades_loss_repa_teacher(self, x, teacher_inputs, y, feat_reg_coeff, log_grad_cosine=False):
        """
        TRADES training with teacher-based REPA for ImageNet.

        Args:
            x: Raw images (B, 3, H, W) for classifier
            teacher_inputs: Either raw images or pre-computed latents for the teacher
            y: Labels (B,)
            feat_reg_coeff: REPA regularization coefficient
            log_grad_cosine: Log cos(grad L_AT, grad L_align) for this batch
        """
        num_timesteps = getattr(self.params, 'num_timesteps', 1)
        patch_wise = getattr(self.params, 'patch_wise_alignment', False)
        mix_fixed_sampled = getattr(self.params, 'mix_fixed_sampled_timesteps', False)

        loss, batch_metrics = trades_loss_repa_teacher(
            self.model,
            x,
            teacher_inputs,
            y,
            self.optimizer,
            self.feature_model,
            teacher_input=getattr(self.params, "teacher_input", "image"),
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            attack=self.params.attack,
            label_smoothing=self.params.ls,
            feat_reg_coeff=feat_reg_coeff,
            beta=self.params.beta,
            num_timesteps=num_timesteps,
            patch_wise=patch_wise,
            mix_fixed_sampled_timesteps=mix_fixed_sampled,
            log_grad_cosine=log_grad_cosine,
        )
        return loss, batch_metrics

    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(
            self.model,
            x,
            y,
            self.optimizer,
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            attack=self.params.attack,
            beta=beta,
            label_smoothing=self.params.ls,
            use_cutmix=self.params.CutMix,
        )
        return loss, batch_metrics

    def trades_loss_consistency(self, x_aug1, x_aug2, y, beta):
        """
        TRADES training with Consistency.
        """
        x = torch.cat([x_aug1, x_aug2], dim=0)
        loss, batch_metrics = trades_loss(
            self.model,
            x,
            y.repeat(2),
            self.optimizer,
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            attack=self.params.attack,
            beta=beta,
            label_smoothing=self.params.ls,
            use_cutmix=self.params.CutMix,
            use_consistency=True,
            cons_lambda=self.params.cons_lambda,
            cons_tem=self.params.cons_tem,
        )
        return loss, batch_metrics

    def trades_loss_LSE(self, x, y, beta):
        """
        TRADES training with LSE loss.
        """
        loss, batch_metrics = trades_loss_LSE(
            self.model,
            x,
            y,
            self.optimizer,
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            attack=self.params.attack,
            beta=beta,
            label_smoothing=self.params.ls,
            clip_value=self.params.clip_value,
            use_cutmix=self.params.CutMix,
            num_classes=self.num_classes,
        )
        return loss, batch_metrics

    def repa_distillation_loss(self, x, y):

        loss, batch_metrics = repa_distillation_loss(
            self.model,
            x,
            y,
            self.feature_model,
            step_size=self.params.attack_step,
            epsilon=self.params.attack_eps,
            perturb_steps=self.params.attack_iter,
            attack=self.params.attack,
        )
        return loss, batch_metrics

    def eval(self, dataloader, adversarial=False, verbose=False):
        """
        Evaluate performance of the model.
        """
        acc = 0.0
        self.wa_model.eval()
        
        iterator = enumerate(dataloader)
        if verbose:
            iterator = enumerate(tqdm(dataloader, desc="Evaluation: "))

        for i, data in iterator:
            if isinstance(data, (list, tuple)) and len(data) == 3:
                x, y, _ = data
            else:
                x, y = data
            x, y = x.to(self.device), y.to(self.device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                with torch.no_grad():
                    out = self.wa_model(x_adv)
            else:
                with torch.no_grad():
                    out = self.wa_model(x)
            acc += accuracy(y, out)
        acc /= len(dataloader)
        return acc

    def save_model(self, path):
        """
        Save model weights.
        """
        # Handle DDP for saving
        model = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(
            {"model_state_dict": self.wa_model.state_dict(), "unaveraged_model_state_dict": model.state_dict()},
            path,
        )

    def save_model_resume(self, path, epoch):
        """
        Save model weights and optimizer.
        """
        # Handle DDP for saving
        model = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(
            {
                "model_state_dict": self.wa_model.state_dict(),
                "unaveraged_model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
            },
            path,
        )

    def load_model(self, path):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if "model_state_dict" not in checkpoint:
            raise RuntimeError("Model weights not found at {}.".format(path))
        self.wa_model.load_state_dict(checkpoint["model_state_dict"])

    def load_model_resume(self, path):
        """
        load model weights and optimizer.
        """
        checkpoint = torch.load(path)
        if "model_state_dict" not in checkpoint:
            raise RuntimeError("Model weights not found at {}.".format(path))
        self.wa_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Handle DDP loading
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint["unaveraged_model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"]


    


def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1.0 + delta) / (10.0 + delta)) if 10.0 + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor

    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked
