import torch

from .resnet import Normalization
from .preact_resnet import preact_resnet
from .resnet import resnet
from .wideresnet import wideresnet
from .diffusion_backbone import diffusion_backbone

from .preact_resnetwithswish import preact_resnetwithswish
from .wideresnetwithswish import wideresnetwithswish
from .wideresnetwithswish_repa_pool import wideresnetwithswish_repa_8x8_pool
from .wideresnetwithswish_noisy import wideresnetwithswish_noisy
from .ti_wideresnetwithswish import ti_wideresnetwithswish
from .swin import swin_t
from .disc_feat_wrapper import DiscFeatExtractorWrapper
from .vit_repa_pool import vit_b_repa_pool, vit_b_token_repa_pool, vit_l_repa_pool, vit_mediumd_rope_repa_pool, vit_b_dinov3_repa_pool
from .convnext_repa_pool import convnext_b_dinov3_repa_pool
from .swin_repa_pool import swin_b_repa_pool, swin_l_repa_pool, swin_b_in22k_repa_pool

from core.data import DATASETS


MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "preact-resnet18",
    "preact-resnet34",
    "preact-resnet50",
    "preact-resnet101",
    "wrn-28-10",
    "wrn-32-10",
    "wrn-34-10",
    "wrn-34-20",
    # "diffusion-backbone-16x16_block0",
    "diffusion-backbone-8x8_block4-uncond-sigma0.1-drop0.0-ft",
    "diffusion-backbone-8x8_block4-uncond-sigma0.1-drop0.0-prob-perturb",
    "diffusion-backbone-8x8_block4-uncond-sigma0.1-drop0.0-prob",
    # "diffusion-backbone-8x8_block4-uncond-sigma0.1-drop0.0-ft-scratch",
    "diffusion-backbone-8x8_block4-uncond-sigma0.1-drop0.0-ft-scratch-perturb",
    # "diffusion-backbone-16x16_block0-perturb",
    "preact-resnet18-swish",
    "preact-resnet34-swish",
    "wrn-28-10-swish",
    "wrn-28-10-swish-repa-8x8",
    "wrn-28-10-swish-repa-8x8-pool",
    "wrn-34-10-swish-repa-8x8-pool",
    "wrn-28-10-swish-repa-8x8-pool-cond",
    "wrn-28-10-swish-repa-16x16-pool-8x8-pool",
    "wrn-28-10-swish-repa-8x8-pool-bn",
    "wrn-28-10-swish-repa-8x8-linear-pool",
    "wrn-34-20-swish",
    "wrn-70-16-swish",
    "wrn-70-16-swish-repa-8x8",
    "wrn-70-16-swish-repa-8x8-pool",
    "wrn-82-8-swish",
    "wrn-82-8-swish-repa-8x8",
    "wrn-82-8-swish-repa-8x8-pool",
    "wrn-28-4-swish",
    "wrn-28-8-swish",
    "wrn-28-8-swish-repa-16x16",
    "wrn-28-8-swish-repa-8x8",
    "wrn-28-8-swish-repa-8x8-pool",
    "wrn-28-10-swish-noisy",
    # "wrn-28-8-swish-repa-8x8-v2",
    # "wrn-28-8-swish-repa-8x8-v2-plus",
    "swin-t",
    "swin-b-repa-pool",
    "swin-l-repa-pool",
    # ImageNet-22K pretrained Swin
    "pre-swin-b-in22k-repa-pool",
    "vit-b-2-repa-pool",
    "vit-b-repa-pool",
    "vit-l-repa-pool",
    "vit-b-repa-patch",
    "vit-b-repa-pool-mlp-2",
    "vit-b-repa-pool-mlp-4",
    "vit-b-repa-pool-mlp-5",
    "vit-b-2-token-repa-pool",
    "vit-b-token-repa-pool",
    "vit-l-repa-pool",
    # Standard IN1K pretrained ViT
    "pre-vit-mediumd-rope-repa-pool",
    # DINOv3 pretrained ViT
    "pre-vit-b-dinov3-repa-pool",
    # DINOv3 pretrained ConvNeXt
    "pre-convnext-b-dinov3-repa-pool",
    # Frozen unconditional EDM encoder + trainable head (cf. App. D)
    "frozen-edm-athead",
]


def create_model(name, normalize, info, device, repa_dim=256, DataParallel=True, use_repa=True):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if info["data"] in ["tiny-imagenet", "tiny-imagenets"]:
        if "wrn" in name and "swish" in name:
            backbone = ti_wideresnetwithswish(name, num_classes=info["num_classes"], device=device)
        else:
            from .ti_preact_resnet import ti_preact_resnet

            backbone = ti_preact_resnet(name, num_classes=info["num_classes"], device=device)

    elif info["data"] in DATASETS and info["data"] not in ["tiny-imagenet", "tiny-imagenets"]:
        if name == "frozen-edm-athead":
            from .frozen_edm_athead import frozen_edm_athead

            backbone = frozen_edm_athead(info, device, num_classes=info["num_classes"])
        elif "preact-resnet" in name and "swish" not in name:
            backbone = preact_resnet(name, num_classes=info["num_classes"], pretrained=False, device=device)
        elif "preact-resnet" in name and "swish" in name:
            backbone = preact_resnetwithswish(name, dataset=info["data"], num_classes=info["num_classes"])
        elif "resnet" in name and "preact" not in name:
            backbone = resnet(name, num_classes=info["num_classes"], pretrained=False, device=device)
        elif "diffusion-backbone" in name:
            backbone = diffusion_backbone(name, num_classes=info["num_classes"], device=device, repa_dim=repa_dim)
        elif "wrn" in name and "swish" not in name:
            backbone = wideresnet(name, num_classes=info["num_classes"], device=device)
        elif "wrn" in name and "swish" in name and "repa" and "8x8-pool" in name:
            if "cond" in name:
                backbone = wideresnetwithswish_repa_8x8_pool(name, num_classes=info["num_classes"], device=device, repa_dim=repa_dim, cond_align_head=True)
            else:
                backbone = wideresnetwithswish_repa_8x8_pool(name, num_classes=info["num_classes"], device=device, repa_dim=repa_dim)
        elif "wrn" in name and "swish" in name and "noisy" in name:
            backbone = wideresnetwithswish_noisy(name, dataset=info["data"], num_classes=info["num_classes"], device=device)
        elif "wrn" in name and "swish" in name:
            backbone = wideresnetwithswish(name, dataset=info["data"], num_classes=info["num_classes"], device=device)
        elif "pre-swin-b-in22k" in name:
            backbone = swin_b_in22k_repa_pool(name, dataset=info["data"], num_classes=info["num_classes"], device=device, repa_dim=repa_dim, use_repa=use_repa)
        elif "pre-vit-mediumd-rope" in name:
            backbone = vit_mediumd_rope_repa_pool(name, dataset=info["data"], num_classes=info["num_classes"], device=device, repa_dim=repa_dim, use_repa=use_repa)
        elif "pre-vit-b-dinov3" in name:
            backbone = vit_b_dinov3_repa_pool(name, dataset=info["data"], num_classes=info["num_classes"], device=device, repa_dim=repa_dim, use_repa=use_repa)
        elif "pre-convnext-b-dinov3" in name:
            backbone = convnext_b_dinov3_repa_pool(name, dataset=info["data"], num_classes=info["num_classes"], device=device, repa_dim=repa_dim, use_repa=use_repa)
        elif "vit-b" in name and "token" in name:
            backbone = vit_b_token_repa_pool(name, dataset=info["data"], num_classes=info["num_classes"], device=device, repa_dim=repa_dim, use_repa=use_repa)
        elif "vit-b" in name:
            backbone = vit_b_repa_pool(name, dataset=info["data"], num_classes=info["num_classes"], device=device, repa_dim=repa_dim, use_repa=use_repa)
        elif "vit-l" in name:
            backbone = vit_l_repa_pool(name, dataset=info["data"], num_classes=info["num_classes"], device=device, repa_dim=repa_dim, use_repa=use_repa)
        else:
            raise ValueError("Invalid model name {}!".format(name))

    else:
        raise ValueError("Models for {} not yet supported!".format(info["data"]))

    if normalize:
        print("Wrapping model with normalization layer")
        model = torch.nn.Sequential(Normalization(info["mean"], info["std"]), backbone)
    else:
        # model = torch.nn.Sequential(backbone)
        model = backbone

    if "diffusion-backbone" in name:
        model = model
        print("Diffusion backbone does not support DataParallel.")
    else:
        if DataParallel:
            model = torch.nn.DataParallel(model)
            print("Using DataParallel.")
    # print(model)
    model = model.to(device)
    return model
