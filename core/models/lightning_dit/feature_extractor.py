"""
LightningDiT Feature Extractor for REPA alignment.

This module provides a wrapper around pre-trained LightningDiT models
for extracting intermediate layer features, useful for:
1. Linear probing to find optimal layer/timestep
2. Feature alignment (REPA) during adversarial training
"""
import os
import yaml
import torch
import torch.nn as nn
from typing import List, Optional, Union
from torch.nn.attention import SDPBackend
from .lightningdit import LightningDiT_models


def sample_timestep_lognorm_like(
    batch_size: int,
    device: torch.device,
    low: float = 0.7,
    high: float = 0.9,
    sigma: float = 1.0,
):
    u = sigma * torch.randn(batch_size, device=device)
    t = torch.sigmoid(u)          # (0, 1)
    t = low + (high - low) * t    # (low, high)
    return t

class LightningDiTFeatureExtractor(nn.Module):
    """
    Feature extractor wrapper for pre-trained LightningDiT.
    
    Can work with pre-computed latents (from PairedImageLatentDataset).
    Returns features in a format compatible with REPA training.
    
    Args:
        config_path: Path to LightningDiT config YAML file
        checkpoint_path: Path to pre-trained checkpoint
        feature_layers: List of layer indices (0-indexed) to extract features from
        timestep: Fixed timestep value in [0, 1], or None for sampling
        timestep_mode: How to sample timesteps if not fixed ('lognorm' or 'uniform')
        timestep_shift: Shift parameter for timestep sampling
        device: Device to load model on
        freeze: Whether to freeze model parameters
        latent_stats_path: Path to latent normalization stats (optional)
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        feature_layers: List[int] = [14],  # Default: middle layer of 28 blocks
        timestep: Optional[float] = None,  # Fixed timestep (0-1), or None for sampling
        timestep_low: float = 0.7,
        timestep_high: float = 0.9,
        device: Union[str, torch.device] = 'cuda',
        freeze: bool = True,
        latent_stats_path: Optional[str] = None,
    ):
        super().__init__()
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_layers = feature_layers
        self.timestep = timestep
        # self.timestep_mode = timestep_mode
        # self.timestep_shift = timestep_shift
        self.timestep_low = timestep_low
        self.timestep_high = timestep_high
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Extract config sections
        model_config = self.config['model']
        vae_config = self.config.get('vae', {})
        data_config = self.config['data']
        
        # Calculate latent size
        downsample_ratio = vae_config.get('downsample_ratio', 16)
        self.latent_size = data_config['image_size'] // downsample_ratio
        self.num_classes = data_config['num_classes']
        
        # Build model
        self.model = LightningDiT_models[model_config['model_type']](
            input_size=self.latent_size,
            num_classes=self.num_classes,
            use_qknorm=model_config.get('use_qknorm', False),
            use_swiglu=model_config.get('use_swiglu', False),
            use_rope=model_config.get('use_rope', False),
            use_rmsnorm=model_config.get('use_rmsnorm', False),
            wo_shift=model_config.get('wo_shift', False),
            in_channels=model_config.get('in_chans', 4),
            learn_sigma=model_config.get('learn_sigma', False),
        )
        
        # Load checkpoint
        print(f"Loading LightningDiT checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema' in checkpoint:
            checkpoint = checkpoint['ema']
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze parameters
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("LightningDiT parameters frozen.")
        else:
            print("WARNING: LightningDiT parameters are NOT frozen.")
        
        # Load latent normalization stats if provided
        self.register_buffer('latent_mean', None)
        self.register_buffer('latent_std', None)
        self.latent_multiplier = data_config.get('latent_multiplier', 1.0)
        
        if latent_stats_path and os.path.exists(latent_stats_path):
            raise ValueError(f"Latent stats should be handled by dataset loader.")
            stats = torch.load(latent_stats_path, map_location='cpu')
            self.register_buffer('latent_mean', stats['mean'].to(self.device))
            self.register_buffer('latent_std', stats['std'].to(self.device))
            print(f"Loaded latent stats from: {latent_stats_path}")
        
        # Print summary
        print(f"LightningDiT Feature Extractor initialized:")
        print(f"  - Model type: {model_config['model_type']}")
        print(f"  - Depth: {self.model.depth} blocks")
        print(f"  - Hidden size: {self.model.hidden_size}")
        print(f"  - Feature layers: {feature_layers}")
        # print(f"  - Timestep: {timestep if timestep is not None else f'sampled ({timestep_mode}, shift={timestep_shift})'}")
        print(f"  - Timestep: {timestep if timestep is not None else f'sampled from ({timestep_low}, {timestep_high})'}")
        print(f"  - Latent size: {self.latent_size}x{self.latent_size}")
    
    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Normalize latents using pre-computed statistics.
        
        Args:
            latents: Input latent tensor (B, C, H, W)
        
        Returns:
            Normalized latents
        """
        if self.latent_mean is not None:
            latents = (latents - self.latent_mean) / self.latent_std
        return latents * self.latent_multiplier
    
    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Denormalize latents back to original scale.
        
        Args:
            latents: Normalized latent tensor
        
        Returns:
            Denormalized latents
        """
        latents = latents / self.latent_multiplier
        if self.latent_mean is not None:
            latents = latents * self.latent_std + self.latent_mean
        return latents
    
    def forward(
        self,
        latents: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        timestep_override: Optional[Union[float, torch.Tensor]] = None,
        return_embed: bool = False,
    ) -> dict:
        """
        Extract features from LightningDiT.
        
        Args:
            latents: (N, C, H, W) pre-encoded latent representations
                     Should already be normalized if using pre-computed latents
            labels: (N,) class labels (optional, uses null class if None)
            timestep_override: Override the configured timestep for this forward pass.
                              Can be a float (same for all samples) or tensor (per-sample timesteps).
            return_embed: Whether to include conditioning embedding in output
            
        Returns:
            dict with:
                - 'features': List of feature tensors from specified layers, each (B, N, D)
                - 'emb': Conditioning embedding (B, D) if return_embed=True
        """
        batch_size = latents.shape[0]
        device = latents.device
        
        # Determine timestep
        t_value = timestep_override if timestep_override is not None else self.timestep
        if t_value is not None:
            if isinstance(t_value, torch.Tensor):
                # Tensor timestep override - use directly (supports per-sample timesteps)
                t = t_value.to(device)
                assert t.shape[0] == batch_size, f"Timestep tensor size {t.shape[0]} != batch size {batch_size}"
            else:
                # Fixed scalar timestep
                t = torch.full((batch_size,), t_value, device=device)
        else:
            # Sample timestep
            t = sample_timestep_lognorm_like(batch_size, device, self.timestep_low, self.timestep_high)
        
        # Handle labels - use null class if not provided
        if labels is None:
            labels = torch.full(
                (batch_size,), 
                self.model.y_embedder.num_classes,
                dtype=torch.long, 
                device=device
            )
        
        # Add noise to latents based on timestep
        noisy_latents = self.get_noisy_latents(latents, t)
        
        # Forward pass with feature extraction
        with torch.no_grad(), torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION]), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            features = self.model.forward_with_only_features(
                noisy_latents, 
                t, 
                labels, 
                return_layers=self.feature_layers
            )
        
        # Ensure we return a dict with 'features' key
        # forward_with_only_features returns a list of feature tensors
        if isinstance(features, list):
            output = {'features': features}
        elif isinstance(features, dict):
            output = features
        else:
            output = {'features': [features]}
        
        # Add embedding if requested (compute conditioning embedding)
        if return_embed:
            raise NotImplementedError("Returning embedding is not implemented yet.")
        
        return output
    
    @property
    def hidden_size(self) -> int:
        """Hidden dimension of the model."""
        return self.model.hidden_size
    
    @property
    def num_patches(self) -> int:
        """Number of patches in the latent representation."""
        return self.model.x_embedder.num_patches
    
    @property
    def cond_embed_dim(self) -> int:
        """Dimension of the conditioning embedding."""
        return self.model.hidden_size
    
    @property
    def depth(self) -> int:
        """Number of transformer blocks."""
        return self.model.depth

    @staticmethod
    def get_noisy_latents(latents: torch.Tensor, timestep: float) -> torch.Tensor:
        """
        Get noisy latents from the model.
        """
        noise = torch.randn_like(latents, device=latents.device)
        t = timestep.view(-1, 1, 1, 1)
        return (1 - t) * noise + t * latents
        # return latents + torch.randn_like(latents) * timestep


def get_lightning_dit_feature_model(
    config_path: str,
    checkpoint_path: str,
    feature_layers: List[int] = [14],
    timestep: Optional[float] = 0.5,
    device: Union[str, torch.device] = 'cuda',
    latent_stats_path: Optional[str] = None,
    freeze: bool = True,
) -> LightningDiTFeatureExtractor:
    """
    Factory function to create LightningDiT feature extractor.
    
    Similar interface to get_feature_model() for EDM, making it easy to swap.
    
    Args:
        config_path: Path to LightningDiT config YAML
        checkpoint_path: Path to pre-trained checkpoint
        feature_layers: List of layer indices to extract features from
        timestep: Fixed timestep value, or None for sampling
        device: Device to load model on
        latent_stats_path: Path to latent normalization stats
        freeze: Whether to freeze model parameters
    
    Returns:
        LightningDiTFeatureExtractor instance
    """
    return LightningDiTFeatureExtractor(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        feature_layers=feature_layers,
        timestep=timestep,
        device=device,
        latent_stats_path=latent_stats_path,
        freeze=freeze,
    )


if __name__ == "__main__":
    # Quick test
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    args = parser.parse_args()
    
    # Create feature extractor
    extractor = LightningDiTFeatureExtractor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        feature_layers=[7, 14, 21],
        timestep=0.5,
    )
    
    # Test with random input
    batch_size = 4
    latent_size = extractor.latent_size
    in_channels = extractor.model.in_channels
    
    dummy_latents = torch.randn(batch_size, in_channels, latent_size, latent_size).cuda()
    dummy_labels = torch.randint(0, 1000, (batch_size,)).cuda()
    
    result = extractor(dummy_latents, dummy_labels, return_embed=True)
    
    print(f"\nTest Results:")
    print(f"  - Number of feature layers: {len(result['features'])}")
    for i, feat in enumerate(result['features']):
        print(f"  - Feature {i} shape: {feat.shape}")
    print(f"  - Embedding shape: {result['emb'].shape}")

