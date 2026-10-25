"""
Frozen-encoder baseline (cf. App. D): frozen (unconditional) EDM encoder + trainable classification head,
trained adversarially (TRADES) with gradients flowing through the frozen encoder.

Protocol notes:
- Unconditional EDM checkpoint, sigma fixed at 0.1, perturb=False (sigma-conditioning
  without added noise) -> the deployed model is fully deterministic, matching the
  App. D "deterministic noise" probe protocol and avoiding obfuscation-by-randomness.
- Only the head is in the optimizer's effective update set (encoder params have
  requires_grad=False); attacks still backprop through the encoder to the input.
"""
import torch
import torch.nn as nn


class FrozenEDMATHead(nn.Module):
    def __init__(self, extractor, num_classes=10, hidden_dim=512):
        super().__init__()
        self.extractor = extractor  # frozen PretrainedFeatureExtractor
        with torch.no_grad():
            dummy = torch.zeros(2, 3, 32, 32, device=next(extractor.parameters()).device)
            feat = extractor(dummy)["features"][0]
        in_dim = feat.shape[1]
        print(f"FrozenEDMATHead: encoder feature {tuple(feat.shape[1:])} -> pooled {in_dim}")
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, repa=False, **kwargs):
        feats = self.extractor(x)["features"][0]  # (B, C, H, W)
        pooled = feats.float().mean(dim=[2, 3])
        logits = self.head(pooled)
        if repa:
            # No REPA head; keep interface compatibility (never used with feat_reg_coeff>0).
            return logits, [pooled.unsqueeze(1)]
        return logits


def frozen_edm_athead(info, device, num_classes=10):
    from edm.get_diffusion import get_feature_model

    extractor = get_feature_model(
        device=device,
        sigma=0.1,
        feature_layer_name=["8x8_block4"],
        unconditional=True,
        perturb=False,
        dataset=info["data"],
    )
    model = FrozenEDMATHead(extractor, num_classes=num_classes)
    return model.to(device)
