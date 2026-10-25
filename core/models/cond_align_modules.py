import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer

class CondAlignHead(nn.Module):
    def __init__(self, input_dim, align_dim, num_classes=10, dropout=0.0):
        super().__init__()
        self.norm = FiLMNorm(input_dim, num_classes)
        self.up_proj = LinearSwiGLU(input_dim, align_dim * 2, bias=False)
        self.middle_layer = LinearSwiGLU(align_dim * 2, align_dim * 2, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(Linear(align_dim * 2, align_dim, bias=True))

    def forward(self, x, cond_norm=None, **kwargs):
        # skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x
        # return x + skip

class FiLMNorm(nn.Module):
    def __init__(self, features, cond_features):
        super().__init__()
        self.linear = Linear(cond_features, features * 2, bias=False)
        self.feature_dim = features

    def forward(self, x, cond):
        B, _, D = x.shape
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        # broadcast scale and shift across all features
        scale = scale.view(B, 1, D)
        shift = scale.view(B, 1, D) 
        return scale * x + shift

class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x):
        return self.linear_swiglu(x, self.weight, self.bias)

    @staticmethod
    def linear_swiglu(x, weight, bias=None):
        x = x @ weight.mT
        if bias is not None:
            x = x + bias
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)

    
class Linear(nn.Linear):
    def forward(self, x):
        return super().forward(x)