"""
LightningDiT module for feature extraction and REPA alignment.
"""

from .lightningdit import (
    LightningDiT,
    LightningDiT_models,
    LightningDiTBlock,
    Attention,
    TimestepEmbedder,
    LabelEmbedder,
    FinalLayer,
)

from .feature_extractor import (
    LightningDiTFeatureExtractor,
    get_lightning_dit_feature_model,
    # sample_timestep,
    sample_timestep_lognorm_like,
)

__all__ = [
    # Model classes
    'LightningDiT',
    'LightningDiT_models',
    'LightningDiTBlock',
    'Attention',
    'TimestepEmbedder',
    'LabelEmbedder',
    'FinalLayer',
    # Feature extraction
    'LightningDiTFeatureExtractor',
    'get_lightning_dit_feature_model',
    # 'sample_timestep',
    'sample_timestep_lognorm_like',
]

