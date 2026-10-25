# Code borrowed from https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py
# (Gowal et al 2020)

from typing import Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.swin_transformer import SwinTransformer
from torch.nn.attention import SDPBackend, sdpa_kernel
from .vit_repa_pool import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD, SVHN_MEAN, SVHN_STD, IMAGENET_MEAN, IMAGENET_STD


class SwinRepaPool(nn.Module):
    """
    Swin model with Repa pooling head.
    Arguments:
        num_classes (int): number of output classes.
        repa_dim (int): dimension of the repa embedding.
    """

    def __init__(
        self,
        num_classes: int = 10,
        repa_dim=256,
        cond_align_head=False,
        pool="avg",
        use_repa=True,
        num_mlp_layers=3,
        # Swin specific args for Swin-T default
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        patch_size=4,
        pretrained_name=None,
    ):
        super().__init__()
        if num_classes == 10:
            mean = CIFAR10_MEAN
            std = CIFAR10_STD
            image_size = 32
            patch_size = 2
            window_size = 4
        elif num_classes == 100:
            mean = CIFAR100_MEAN
            std = CIFAR100_STD
            image_size = 32
            patch_size = 2
            window_size = 4
        elif num_classes == 1000:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
            image_size = 224
        else:
            raise ValueError(f"Number of classes not implemented except for CIFAR10/100, and ImageNet.")
        
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        
        # Use pretrained timm model if specified
        if pretrained_name is not None:
            print(f"Loading pretrained model: {pretrained_name}")
            self.swin = timm.create_model(pretrained_name, pretrained=True, num_classes=num_classes)
        else:
            self.swin = SwinTransformer(
                img_size=image_size,
                patch_size=patch_size,
                in_chans=3,
                num_classes=num_classes,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                global_pool=pool,
            )
        
        self.repa_dim = repa_dim
        repa_input_dim = self.swin.num_features
        
        if cond_align_head:
             raise NotImplementedError("CondAlignHead is not implemented for Swin")
        else:
            if use_repa:
                bottleneck_dim = repa_dim * 2
                print(f"Using dimension {repa_input_dim} -> {bottleneck_dim} -> {repa_dim} for Repa head")
                self.repa_head = self.build_repa_head(repa_input_dim, bottleneck_dim, repa_dim, num_mlp_layers=num_mlp_layers)
                print(f"Using {num_mlp_layers} MLP layers for Repa head")
            else:
                self.repa_head = None
        self.repa_input_dim = repa_input_dim

    def forward_features(self, x):
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
            if self.mean.device != x.device:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
            
            x = (x - self.mean) / self.std
            x = self.swin.forward_features(x)
            x = self.swin.forward_head(x, pre_logits=True)
        return x

    def forward_features_and_logits(self, x):
        x = self.forward_features(x)
        logits = self.swin.head.flatten(self.swin.head.fc(x))
        return x, logits

    def forward(self, x, repa=False):
        x, logits = self.forward_features_and_logits(x)
        if repa:
            repa_out = self.repa_head(x)
            return logits, [repa_out.view(x.size(0), -1, self.repa_dim)]
        return logits

    def build_repa_head(self, hidden_size, projector_dim, z_dim, num_mlp_layers=3):
        layers = []
        layers.append(nn.Linear(hidden_size, projector_dim))
        layers.append(nn.SiLU())
        for _ in range(num_mlp_layers - 2):
            layers.append(nn.Linear(projector_dim, projector_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(projector_dim, z_dim))
        return nn.Sequential(*layers)

def swin_b_repa_pool(name, dataset="cifar10", num_classes=10, device="cuda", repa_dim=256, cond_align_head=False, use_repa=True):
    return SwinRepaPool(
        num_classes=num_classes,
        repa_dim=repa_dim,
        cond_align_head=cond_align_head,
        pool="avg",
        use_repa=use_repa,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
    )

def swin_l_repa_pool(name, dataset="cifar10", num_classes=10, device="cuda", repa_dim=256, cond_align_head=False, use_repa=True):
    return SwinRepaPool(
        num_classes=num_classes,
        repa_dim=repa_dim,
        cond_align_head=cond_align_head,
        pool="avg",
        use_repa=use_repa,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
    )

def swin_b_in22k_repa_pool(name, dataset="cifar10", num_classes=10, device="cuda", repa_dim=256, cond_align_head=False, use_repa=True):
    """
    Swin Base pretrained on ImageNet-22K.
    Uses pretrained timm model: swin_base_patch4_window7_224.ms_in22k
    Model specs: embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32)
    """
    name_parts = name.split("-")
    if "mlp" in name:
        mlp_idx = name_parts.index("mlp")
        num_mlp_layers = int(name_parts[mlp_idx + 1])
    else:
        num_mlp_layers = 3
    
    return SwinRepaPool(
        num_classes=num_classes,
        repa_dim=repa_dim,
        cond_align_head=cond_align_head,
        pool="avg",
        use_repa=use_repa,
        num_mlp_layers=num_mlp_layers,
        pretrained_name="swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    )

