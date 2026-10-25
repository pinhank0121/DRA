import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F

# Add the parent directory to the path to import from edm
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../edm/dnnlib'))
from edm.get_diffusion import PretrainedFeatureExtractor

class DiffusionBackbone(nn.Module):
    def __init__(self, 
                 network_pkl="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl", 
                 feature_layer_names=["8x8_block4"],
                 num_classes=10, 
                 device="cuda", 
                 sigma=None, 
                 unconditional=False, 
                 perturb=False,
                 cfg=False,
                 freeze=True,
                 dropout_rate=0.0,
                 from_scratch=False,
                 **kwargs,
                 ):
        super(DiffusionBackbone, self).__init__()
        
        # Initialize the feature extractor
        if unconditional:
            network_pkl = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl"

        self.feature_extractor = PretrainedFeatureExtractor(
            network_pkl=network_pkl,
            device=device,
            feature_layer_names=feature_layer_names,
            sigma=sigma,
            unconditional=unconditional,
            perturb=perturb,
            cfg=cfg,
            freeze=freeze,
            from_scratch=from_scratch
        )
        
        # Determine the feature dimension
        # Default output is 256x8x8 for the "8x8_block4" layer
        self.feature_dim = 256
        # self.feature_size = 16
        
        # Create the classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, num_classes)
        )
        self.freeze = freeze
        
        # Freeze the feature extractor
        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False
        
    def forward(self, x, class_labels=None):
        # Extract features from the diffusion model
        # x = x * 2.0 - 1.0
        # if self.freeze:
        #     features = self.feature_extractor(x, class_labels)
        # else:
        features = self.feature_extractor(x, class_labels)
        
        # if it is tuple throw away the second element
        if isinstance(features, tuple):
            if len(features) > 2:
                raise ValueError(f"Expected tuple of length 2, got {len(features)}")
            features = features[0]
        
        # If multiple feature layers are used, only use the first one
        # This could be modified to combine multiple features if needed
        if isinstance(features, dict):
            features = features["features"]
        if isinstance(features, list):
            features = features[0]
        
        # Pass features through the classification head
        output = self.head(features)
        
        return output
    
    def forward_features(self, x, class_labels=None):
        features = self.feature_extractor(x, class_labels)
        # if it is tuple throw away the second element
        if isinstance(features, tuple):
            if len(features) > 2:
                raise ValueError(f"Expected tuple of length 2, got {len(features)}")
            features = features[0]
        # If multiple feature layers are used, only use the first one
        if isinstance(features, dict):
            features = features["features"]
        if isinstance(features, list):
            features = features[0]

        # return features
        return F.adaptive_avg_pool2d(features, (1, 1))


def diffusion_backbone(name, num_classes=10, device="cuda", **kwargs):
    """
    Create a model with a diffusion backbone.
    
    Args:
        name (str): Model name in format 'diffusion-backbone-{feature_layer_name}'
        num_classes (int): Number of output classes
        device (str): Device to use
        
    Returns:
        DiffusionBackbone: The model with diffusion backbone
    """
    # Parse model name to get configuration
    # Expected format: diffusion-backbone-{feature_layer_name}
    # Example: diffusion-backbone-16x16_block0
    parts = name.split('-')
    
    # Default settings
    unconditional = False
    perturb = False
    sigma = None
    dropout_rate = 0.0
    freeze = True
    from_scratch = False
    
    # Parse the feature layer name and additional settings from the model name
    if len(parts) >= 3:
        feature_layer_name = parts[2]
        
        # Check for additional settings
        if len(parts) > 3:
            for part in parts[3:]:
                if part == "uncond":
                    unconditional = True
                elif part == "perturb":
                    perturb = True
                elif part == "ft":
                    freeze = False
                elif part == "prob":
                    freeze = True
                elif part == "scratch":
                    from_scratch = True
                elif part.startswith("sigma"):
                    try:
                        sigma = float(part.replace("sigma", ""))
                    except ValueError:
                        print(f"Warning: Could not parse sigma value from {part}")
                elif part.startswith("drop"):
                    try:
                        dropout_rate = float(part.replace("drop", ""))
                    except ValueError:
                        print(f"Warning: Could not parse dropout rate from {part}")
                else:
                    raise ValueError(f"Unknown part in model name: {part}")
    else:
        # Default feature layer name if not specified
        feature_layer_name = "8x8_block4"
    
    # You can add more feature layers if needed
    feature_layer_names = [feature_layer_name]
    # print(f"Using feature layer: {feature_layer_names}")
    
    # Create and return the model
    return DiffusionBackbone(
        feature_layer_names=feature_layer_names,
        num_classes=num_classes,
        device=device,
        unconditional=unconditional,
        perturb=perturb,
        sigma=sigma,
        freeze=freeze,
        dropout_rate=dropout_rate,
        from_scratch=from_scratch,
        **kwargs
    )