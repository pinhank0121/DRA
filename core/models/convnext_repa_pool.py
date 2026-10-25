# ConvNeXt model with REPA pooling head using DINOv3 pretrained weights

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.nn.attention import SDPBackend, sdpa_kernel
from .vit_repa_pool import CIFAR10_MEAN, CIFAR10_STD, CIFAR100_MEAN, CIFAR100_STD, IMAGENET_MEAN, IMAGENET_STD


class ConvNeXtRepaPool(nn.Module):
    """
    ConvNeXt model with Repa pooling head.
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
        pretrained_name=None,
    ):
        super().__init__()
        if num_classes == 10:
            mean = CIFAR10_MEAN
            std = CIFAR10_STD
        elif num_classes == 100:
            mean = CIFAR100_MEAN
            std = CIFAR100_STD
        elif num_classes == 1000:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
        else:
            raise ValueError(f"Number of classes not implemented except for CIFAR10/100, and ImageNet.")
        
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        
        # Use pretrained timm model
        if pretrained_name is not None:
            print(f"Loading pretrained model: {pretrained_name}")
            self.convnext = timm.create_model(pretrained_name, pretrained=True, num_classes=num_classes)
            embed_dim = self.convnext.num_features
        else:
            raise ValueError("ConvNeXt requires a pretrained_name to be specified")
        
        self.repa_dim = repa_dim
        repa_input_dim = embed_dim
        
        if cond_align_head:
            raise NotImplementedError("CondAlignHead is not implemented for ConvNeXt")
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
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = (x - self.mean) / self.std
            x = self.convnext.forward_features(x)
            return x

    def forward_features_and_logits(self, x):
        x = self.forward_features(x)
        features = self.convnext.forward_head(x, pre_logits=True)
        logits = self.convnext.head.fc(features)
        return features, logits

    def forward(self, x, repa=False):
        x, logits = self.forward_features_and_logits(x)
        if repa:
            assert len(x.shape) == 2, f"x.shape should be (batch_size, hidden_size) but got {x.shape}"
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


def convnext_b_dinov3_repa_pool(name, dataset="cifar10", num_classes=10, device="cuda", repa_dim=256, cond_align_head=False, use_repa=True):
    """
    ConvNeXt Base with DINOv3 pretrained weights.
    Uses pretrained timm model: convnext_base.dinov3_lvd1689m
    Model specs: embed_dim=1024
    """
    name_parts = name.split("-")
    if "mlp" in name:
        mlp_idx = name_parts.index("mlp")
        num_mlp_layers = int(name_parts[mlp_idx + 1])
    else:
        num_mlp_layers = 3
    
    return ConvNeXtRepaPool(
        num_classes=num_classes,
        repa_dim=repa_dim,
        cond_align_head=cond_align_head,
        pool="avg",
        use_repa=use_repa,
        num_mlp_layers=num_mlp_layers,
        pretrained_name="convnext_base.dinov3_lvd1689m",
    )

