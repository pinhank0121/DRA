import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SSLTeacherFeatureExtractor(nn.Module):
    """
    Frozen self-supervised teacher wrapper for ImageNet REPA.

    Returns a dict with a single `features` entry so it matches the
    LightningDiT extractor contract already used by the training code.
    """

    DEFAULT_MODELS = {
        "dinov3": "vit_base_patch16_dinov3_qkvb.lvd1689m",
        "mae": "hf-hub:timm/vit_large_patch16_224.mae",
        "clip": "ViT-B/16",
    }

    def __init__(
        self,
        teacher_type,
        model_name=None,
        pool="avg",
        device="cuda",
        freeze=True,
        standardize=False,
    ):
        super().__init__()
        self.teacher_type = teacher_type
        self.model_name = model_name or self.DEFAULT_MODELS[teacher_type]
        self.pool = pool
        self.standardize = standardize
        self.device = torch.device(device) if isinstance(device, str) else device
        self.clip_backend = None

        if teacher_type in {"dinov3", "mae"}:
            self.model = timm.create_model(self.model_name, pretrained=True)
            self.model.to(self.device)
            self.model.eval()

            pretrained_cfg = getattr(self.model, "pretrained_cfg", {}) or {}
            input_size = pretrained_cfg.get("input_size", (3, 224, 224))
            self.input_size = int(input_size[-1])
            self.register_buffer(
                "mean",
                torch.tensor(pretrained_cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor(pretrained_cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1),
                persistent=False,
            )
            self.hidden_size = getattr(self.model, "num_features", None) or getattr(self.model, "embed_dim", None)
        elif teacher_type == "clip":
            if self.model_name.startswith("vit_") or self.model_name.startswith("hf-hub:"):
                self.clip_backend = "timm"
                self.model = timm.create_model(self.model_name, pretrained=True)
                self.model.to(self.device)
                self.model.eval()

                pretrained_cfg = getattr(self.model, "pretrained_cfg", {}) or {}
                input_size = pretrained_cfg.get("input_size", (3, 224, 224))
                self.input_size = int(input_size[-1])
                self.register_buffer(
                    "mean",
                    torch.tensor(pretrained_cfg.get("mean", (0.48145466, 0.4578275, 0.40821073))).view(1, 3, 1, 1),
                    persistent=False,
                )
                self.register_buffer(
                    "std",
                    torch.tensor(pretrained_cfg.get("std", (0.26862954, 0.26130258, 0.27577711))).view(1, 3, 1, 1),
                    persistent=False,
                )
                self.hidden_size = getattr(self.model, "num_features", None) or getattr(self.model, "embed_dim", None)
            else:
                self.clip_backend = "openai"
                try:
                    import clip
                except ImportError as exc:
                    raise ImportError(
                        "CLIP teacher support requires the `clip` package to be installed."
                    ) from exc

                self.model, _ = clip.load(self.model_name, device="cpu", jit=False)
                self.model.to(self.device)
                self.model.eval()
                self.input_size = int(self.model.visual.input_resolution)
                self.register_buffer(
                    "mean",
                    torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1),
                    persistent=False,
                )
                self.register_buffer(
                    "std",
                    torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1),
                    persistent=False,
                )
                self.hidden_size = int(self.model.visual.output_dim)
        else:
            raise ValueError(f"Unsupported teacher_type: {teacher_type}")

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if self.hidden_size is None:
            raise ValueError(f"Could not infer hidden size for teacher model {self.model_name}.")

        if self.teacher_type == "clip" and self.pool == "patch":
            raise ValueError("CLIP teacher only supports pooled features (`avg` or `cls`).")

        # if self.pool == "pre-logits" and self.teacher_type not in {"dinov3", "mae"}:
        #     raise ValueError("Pre-logits pooling is only supported for dinov3 and mae teachers.")

    def _preprocess_images(self, x):
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _as_feature_dict(self, features):
        return {"features": [features]}

    def _pool_vit_tokens(self, tokens):
        num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        patch_tokens = tokens[:, num_prefix_tokens:] if tokens.size(1) > num_prefix_tokens else tokens

        if self.pool == "patch":
            return patch_tokens
        if self.pool == "cls" and num_prefix_tokens > 0:
            return tokens[:, :1]

        pooled = patch_tokens.mean(dim=1, keepdim=True)
        return pooled

    def _extract_timm_features(self, x):
        features = self.model.forward_features(x)
        # sanity check to see if the implementation matter here
        # NOTE: remember to comment this out when not needed
        if self.teacher_type in {"dinov3", "mae"} and self.pool == "pre-logits":
            features = self.model.forward_head(features, pre_logits=True)

        if isinstance(features, dict):
            if self.pool == "patch":
                patch_tokens = features.get("x_norm_patchtokens")
                if patch_tokens is not None:
                    return patch_tokens
            if self.pool == "cls":
                cls_token = features.get("x_norm_clstoken")
                if cls_token is not None:
                    return cls_token.unsqueeze(1)
            if "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"].mean(dim=1, keepdim=True)
            if "x_prenorm" in features:
                return self._extract_timm_features_from_tensor(features["x_prenorm"])
            raise ValueError(f"Unsupported timm feature dict keys: {sorted(features.keys())}")

        return self._extract_timm_features_from_tensor(features)

    def _extract_timm_features_from_tensor(self, features):
        if features.dim() == 4:
            # (B, C, H, W) -> (B, H * W, C)
            features = features.flatten(2).transpose(1, 2)
            if self.pool == "patch":
                return features
            return features.mean(dim=1, keepdim=True)

        if features.dim() == 3:
            return self._pool_vit_tokens(features)

        if features.dim() == 2:
            return features.unsqueeze(1)

        raise ValueError(f"Unsupported feature tensor shape: {tuple(features.shape)}")

    def forward(self, x, labels=None, timestep_override=None, return_embed=False):
        if timestep_override is not None:
            raise ValueError(f"{self.teacher_type} teacher does not support timestep overrides.")
        if return_embed:
            raise NotImplementedError("return_embed is not implemented for SSL teachers.")

        x = self._preprocess_images(x)

        autocast_enabled = self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            if self.teacher_type == "clip" and self.clip_backend == "openai":
                features = self.model.encode_image(x).unsqueeze(1)
            else:
                features = self._extract_timm_features(x)

        if self.standardize:
            # Per-channel standardization using batch statistics over (batch, tokens).
            feat_mean = features.mean(dim=(0, 1), keepdim=True)
            feat_std = features.std(dim=(0, 1), keepdim=True)
            features = (features - feat_mean) / (feat_std + 1e-6)

        return self._as_feature_dict(features)
