# Code borrowed from https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py
# (Gowal et al 2020)

from typing import Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cond_align_modules import CondAlignHead


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
SVHN_MEAN = (0.5, 0.5, 0.5)
SVHN_STD = (0.5, 0.5, 0.5)

_ACTIVATION = {
    "relu": nn.ReLU,
    "swish": nn.SiLU,
}


class _Block(nn.Module):
    """
    WideResNet Block.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        activation_fn (nn.Module): activation function.
    """

    def __init__(self, in_planes, out_planes, stride, activation_fn=nn.ReLU):
        super().__init__()
        self.batchnorm_0 = nn.BatchNorm2d(in_planes, momentum=0.01)
        self.relu_0 = activation_fn(inplace=True)
        # We manually pad to obtain the same effect as `SAME` (necessary when `stride` is different than 1).
        self.conv_0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.batchnorm_1 = nn.BatchNorm2d(out_planes, momentum=0.01)
        self.relu_1 = activation_fn(inplace=True)
        self.conv_1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        else:
            self.shortcut = None
        self._stride = stride

    def forward(self, x):
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))
        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError("Unsupported `stride`.")
        out = self.conv_0(v)
        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)
        out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _BlockGroup(nn.Module):
    """
    WideResNet block group.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        activation_fn (nn.Module): activation function.
    """

    def __init__(self, num_blocks, in_planes, out_planes, stride, activation_fn=nn.ReLU):
        super().__init__()
        block = []
        for i in range(num_blocks):
            block.append(
                _Block(
                    i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, activation_fn=activation_fn
                )
            )
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class WideResNetRepa8x8Pool(nn.Module):
    """
    WideResNet model
    Arguments:
        num_classes (int): number of output classes.
        depth (int): number of layers.
        width (int): width factor.
        activation_fn (nn.Module): activation function.
        mean (tuple): mean of dataset.
        std (tuple): standard deviation of dataset.
        padding (int): padding.
        num_input_channels (int): number of channels in the input.
        cond_align_head (nn.Module): conditional alignment head.
    """

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        width: int = 10,
        activation_fn: nn.Module = nn.ReLU,
        mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
        std: Union[Tuple[float, ...], float] = CIFAR10_STD,
        padding: int = 0,
        num_input_channels: int = 3,
        repa_dim=256,
        cond_align_head=False,
    ):
        super().__init__()
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        self.padding = padding
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.init_conv = nn.Conv2d(num_input_channels, num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer = nn.Sequential(
            _BlockGroup(num_blocks, num_channels[0], num_channels[1], 1, activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2, activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2, activation_fn=activation_fn),
        )
        self.batchnorm = nn.BatchNorm2d(num_channels[3], momentum=0.01)
        self.relu = activation_fn(inplace=True)
        self.logits = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]
        self.num_classes = num_classes

        self.repa_dim = repa_dim
        repa_input_dim = num_channels[3]
        self.cond_align_head = cond_align_head
        print(f"cond_align_head: {cond_align_head}")
        if cond_align_head:
            self.repa_head = CondAlignHead(repa_input_dim, repa_dim, num_classes)
        else:
            self.repa_head = build_repa_head(repa_input_dim, repa_dim * 2, repa_dim)
        self.repa_input_dim = repa_input_dim

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    # @torch.compile
    def forward(self, x, repa=False, cond_embed=None):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        if self.mean_cuda is None:
            self.mean_cuda = self.mean.cuda()
            self.std_cuda = self.std.cuda()
        out = (x - self.mean_cuda) / self.std_cuda

        out = self.init_conv(out)
        bs = x.shape[0]
        if isinstance(repa, torch.fx.Proxy):  # Check if it's a proxy
            repa = False  # Convert to a dummy tensor value
        if repa:
            out = self.layer(out)  # should be 8x8 spatial size
            if self.cond_align_head:
                repa_out = self.repa_head(F.avg_pool2d(out, 8).permute(0, 2, 3, 1).view(bs, -1, self.repa_input_dim), cond_embed)
            else:
                repa_out = self.repa_head(F.avg_pool2d(out, 8).permute(0, 2, 3, 1).view(bs, -1, self.repa_input_dim))

            out = self.relu(self.batchnorm(out))
            out = F.avg_pool2d(out, 8)
            out = out.view(-1, self.num_channels)

            return self.logits(out), [repa_out]

        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)

    def forward_features(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)

        if self.mean_cuda is None:
            self.mean_cuda = self.mean.cuda()
            self.std_cuda = self.std.cuda()
        out = (x - self.mean_cuda) / self.std_cuda

        out = self.init_conv(out)

        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        # out = out.view(-1, self.num_channels)
        return out

def build_repa_head(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
        nn.Linear(hidden_size, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, projector_dim),
        nn.SiLU(),
        nn.Linear(projector_dim, z_dim),
    )


def wideresnetwithswish_repa_8x8_pool(name, dataset="cifar10", num_classes=10, device="cpu", repa_dim=256, cond_align_head=False):
    """
    Returns suitable Wideresnet model with Swish activation function from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        device (str or torch.device): device to work on.
        dataset (str): dataset to use.
        cond_align_head (bool): whether to use conditional alignment head.
    Returns:
        torch.nn.Module.
    """
    name_parts = name.split("-")
    depth = int(name_parts[1])
    widen = int(name_parts[2])
    act_fn = name_parts[3]

    print(f"(REPA+) WideResNet-{depth}-{widen}-{act_fn} align post last normalization layer.")
    print("3-Layer MLP for repa head.")
    if "cifar100" in dataset:
        return WideResNetRepa8x8Pool(
            num_classes=num_classes,
            depth=depth,
            width=widen,
            activation_fn=_ACTIVATION[act_fn],
            mean=CIFAR100_MEAN,
            std=CIFAR100_STD,
            repa_dim=repa_dim,
            cond_align_head=cond_align_head,
        )
    elif "svhn" in dataset:
        return WideResNetRepa8x8Pool(
            num_classes=num_classes,
            depth=depth,
            width=widen,
            activation_fn=_ACTIVATION[act_fn],
            mean=SVHN_MEAN,
            std=SVHN_STD,
            repa_dim=repa_dim,
            cond_align_head=cond_align_head,
        )
    else:
        return WideResNetRepa8x8Pool(
            num_classes=num_classes,
            depth=depth,
            width=widen,
            activation_fn=_ACTIVATION[act_fn],
            repa_dim=repa_dim,
            cond_align_head=cond_align_head,
        )
