# Code borrowed from https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py
# (Gowal et al 2020)

from typing import Optional, Tuple, Union
import types

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cond_align_modules import CondAlignHead

import timm
from timm.models.vision_transformer import VisionTransformer
from torch.nn.attention import SDPBackend, sdpa_kernel

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
SVHN_MEAN = (0.5, 0.5, 0.5)
SVHN_STD = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_ACTIVATION = {
    "relu": nn.ReLU,
    "swish": nn.SiLU,
}


def _patch_eva_attention_inplace_ops(model):
    """
    Patch EVA attention modules to replace in-place operations with non-in-place ones.
    This fixes gradient computation issues during adversarial training (TRADES).
    The issue is in timm's EvaAttention.forward: `qkv += qkv_bias` (in-place op).
    """
    try:
        from timm.models.eva import EvaAttention
        from timm.layers.pos_embed_sincos import apply_rot_embed_cat
    except ImportError:
        return model
    
    def patched_forward(self, x, rope: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        B, N, C = x.shape

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
                if self.qkv_bias_separate:
                    qkv = self.qkv(x)
                    qkv = qkv + qkv_bias  # Fixed: changed from qkv += qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)

        if rope is not None:
            npt = self.num_prefix_tokens
            q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope)], dim=2).type_as(v)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            if attn_mask is not None:
                attn_mask = attn_mask.to(torch.bool)
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    # Patch all EvaAttention modules in the model
    patched_count = 0
    for module in model.modules():
        if isinstance(module, EvaAttention):
            module.forward = types.MethodType(patched_forward, module)
            patched_count += 1
    
    if patched_count > 0:
        print(f"Patched {patched_count} EVA attention modules to fix in-place operations")
    return model


class ViTRepaPool(nn.Module):
    """
    ViT model with Repa pooling head.
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
        version="base",
        num_mlp_layers=3,
        patch_wise_alignment=False,
        pretrained_name=None,
    ):
        super().__init__()
        if num_classes == 10:
            mean = CIFAR10_MEAN
            std = CIFAR10_STD
            image_size = 32
            patch_size = 2
        elif num_classes == 100:
            mean = CIFAR100_MEAN
            std = CIFAR100_STD
            image_size = 32
            patch_size = 2
        elif num_classes == 1000:
            mean = IMAGENET_MEAN
            std = IMAGENET_STD
            image_size = 224
            patch_size = 14
        else:
            raise ValueError(f"Number of classes not implemented except for CIFAR10/100, and ImageNet.")
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.patch_wise_alignment = patch_wise_alignment
        fc_norm = False if pool == "avg" else None
        class_token = True if pool == "token" else False
        
        # Use pretrained timm model if specified
        if pretrained_name is not None:
            print(f"Loading pretrained model: {pretrained_name}")
            # For DINOv3 models, use their native 256x256 resolution
            # if "dinov3" in pretrained_name:
            #     self.vit = timm.create_model(pretrained_name, pretrained=True, num_classes=num_classes, img_size=256)
            # else:
            self.vit = timm.create_model(pretrained_name, pretrained=True, num_classes=num_classes)
            embed_dim = self.vit.embed_dim
        else:
            if version == "base":
                embed_dim = 768
                depth = 12
                num_heads = 16
            elif version == "large":
                embed_dim = 1024
                depth = 24
                num_heads = 16
            else:
                raise ValueError(f"Version {version} not implemented.")
            self.vit = VisionTransformer(
                img_size=image_size,
                patch_size=patch_size,
                num_classes=num_classes,
                global_pool=pool,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                class_token=class_token,
                fc_norm=fc_norm,
            )
        self.repa_dim = repa_dim
        repa_input_dim = embed_dim
        
        if cond_align_head:
            raise NotImplementedError("CondAlignHead is not implemented for ViT")
            # self.repa_head = CondAlignHead(repa_input_dim, repa_dim, num_classes)
        else:
            if use_repa:
                # if repa_dim >= repa_input_dim:
                #     bottleneck_dim = repa_dim
                # else:
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
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            x = (x - self.mean) / self.std
            x = self.vit.forward_features(x)
            # x = self.vit.forward_head(x, pre_logits=True)
            return x

    # @torch.compile
    def forward_features_and_logits(self, x):
        x = self.forward_features(x)
        if self.patch_wise_alignment:
            features = x
            logits = self.vit.forward_head(features, pre_logits=False)
        else:
            features = self.vit.forward_head(x, pre_logits=True)
            logits = self.vit.head(features)
        
        return features, logits

    def forward(self, x, repa=False):
        # x = self.forward_features(x)
        # logits = self.vit.head(x)
        x, logits = self.forward_features_and_logits(x)
        if repa:
            assert len(x.shape) == 2, f"x.shape should be (batch_size, hidden_size) but got {x.shape}"
            repa_out = self.repa_head(x)
            return logits, [repa_out.view(x.size(0), -1, self.repa_dim)]
        return logits

    def build_repa_head(self, hidden_size, projector_dim, z_dim, num_mlp_layers=3):
        if self.patch_wise_alignment:
            return nn.Conv2d(hidden_size, z_dim, kernel_size=3, padding=1)
        else:
            layers = []
            layers.append(nn.Linear(hidden_size, projector_dim))
            layers.append(nn.SiLU())
            for _ in range(num_mlp_layers - 2):
                layers.append(nn.Linear(projector_dim, projector_dim))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(projector_dim, z_dim))
            return nn.Sequential(*layers)

def vit_b_repa_pool(name, dataset="cifar10", num_classes=10, device="cuda", repa_dim=256, cond_align_head=False, use_repa=True):
    # find number of mlp layers in mlp in the name
    name_parts = name.split("-")
    if "mlp" in name:
        assert name_parts[-2] == "mlp"
        num_mlp_layers = int(name_parts[-1])
    else:
        num_mlp_layers = 3
    return ViTRepaPool(
        num_classes=num_classes,
        repa_dim=repa_dim,
        cond_align_head=cond_align_head,
        pool="avg",
        use_repa=use_repa,
        num_mlp_layers=num_mlp_layers,
    )

def vit_l_repa_pool(name, dataset="cifar10", num_classes=10, device="cuda", repa_dim=256, cond_align_head=False, use_repa=True):
    name_parts = name.split("-")
    if "mlp" in name:
        assert name_parts[4] == "mlp"
        num_mlp_layers = int(name_parts[5])
    else:
        num_mlp_layers = 3
    return ViTRepaPool(
        num_classes=num_classes,
        repa_dim=repa_dim,
        cond_align_head=cond_align_head,
        pool="avg",
        use_repa=use_repa,
        version="large",
        num_mlp_layers=num_mlp_layers,
    )

def vit_b_token_repa_pool(name, dataset="cifar10", num_classes=10, device="cuda", repa_dim=256, cond_align_head=False, use_repa=True):
    name_parts = name.split("-")
    if "mlp" in name:
        assert name_parts[4] == "mlp"
        num_mlp_layers = int(name_parts[5])
    else:
        num_mlp_layers = 3
    if "patch" in name:
        patch_wise_alignment = True
    else:
        patch_wise_alignment = False

    return ViTRepaPool(
        num_classes=num_classes,
        repa_dim=repa_dim,
        cond_align_head=cond_align_head,
        pool="token",
        use_repa=use_repa,
        num_mlp_layers=num_mlp_layers,
        patch_wise_alignment=patch_wise_alignment,
    )

def vit_mediumd_rope_repa_pool(name, dataset="cifar10", num_classes=10, device="cuda", repa_dim=256, cond_align_head=False, use_repa=True):
    """
    ViT MediumD with ROPE (Rotary Position Embeddings) and registers.
    Uses pretrained timm model: vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k
    Model specs: embed_dim=512, depth=20, num_heads=8, image_size=256, patch_size=16
    """
    name_parts = name.split("-")
    if "mlp" in name:
        # Find mlp in name parts
        mlp_idx = name_parts.index("mlp")
        num_mlp_layers = int(name_parts[mlp_idx + 1])
    else:
        num_mlp_layers = 3
    
    model = ViTRepaPool(
        num_classes=num_classes,
        repa_dim=repa_dim,
        cond_align_head=cond_align_head,
        pool="avg",
        use_repa=use_repa,
        num_mlp_layers=num_mlp_layers,
        pretrained_name="vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k",
    )
    
    # Patch EVA attention to fix in-place operation issues with adversarial training
    _patch_eva_attention_inplace_ops(model)
    
    return model


def vit_b_dinov3_repa_pool(name, dataset="cifar10", num_classes=10, device="cuda", repa_dim=256, cond_align_head=False, use_repa=True):
    """
    ViT Base with DINOv3 pretrained weights.
    Uses pretrained timm model: vit_base_patch16_dinov3_qkvb.lvd1689m
    Model specs: embed_dim=768, depth=12, num_heads=12, image_size=256, patch_size=16
    """
    name_parts = name.split("-")
    if "mlp" in name:
        # Find mlp in name parts
        mlp_idx = name_parts.index("mlp")
        num_mlp_layers = int(name_parts[mlp_idx + 1])
    else:
        num_mlp_layers = 3
    
    model = ViTRepaPool(
        num_classes=num_classes,
        repa_dim=repa_dim,
        cond_align_head=cond_align_head,
        pool="avg",
        use_repa=use_repa,
        num_mlp_layers=num_mlp_layers,
        pretrained_name="vit_base_patch16_dinov3_qkvb.lvd1689m",
    )
    
    # Patch EVA attention to fix in-place operation issues with adversarial training
    # _patch_eva_attention_inplace_ops(model)
    
    return model