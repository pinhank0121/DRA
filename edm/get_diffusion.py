import os
import sys
sys.path.append(os.path.dirname(__file__))
import dnnlib
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.functional import silu
from types import MethodType
import numpy as np

def new_net_forward(
    self,
    x,
    sigma,
    class_labels=None,
    return_feature_from=None,
    force_fp32=False,
    **model_kwargs,
):
    assert return_feature_from is not None
    x = x.to(torch.float32)
    sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
    class_labels = (
        None
        if self.label_dim == 0
        else (
            torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
    )
    dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

    # c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
    # c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
    c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
    c_noise = sigma.log() / 4

    features = self.model(
        (c_in * x).to(dtype),
        c_noise.flatten(),
        class_labels=class_labels,
        return_feature_from=return_feature_from,
        **model_kwargs,
    )
    return features
    # assert F_x.dtype == dtype
    # D_x = c_skip * x + c_out * F_x.to(torch.float32)
    # return D_x


def new_model_forward(self, x, noise_labels, class_labels, augment_labels=None, return_feature_from=None, return_embed=False):
    # Mapping.
    emb = self.map_noise(noise_labels)
    emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
    if self.map_label is not None:
        tmp = class_labels
        if self.training and self.label_dropout:
            tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
        emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
    if self.map_augment is not None and augment_labels is not None:
        emb = emb + self.map_augment(augment_labels)
    emb = silu(self.map_layer0(emb))
    emb = silu(self.map_layer1(emb))

    # Encoder.
    skips = []
    aux = x
    return_features = []
    for name, block in self.enc.items():
        if "aux_down" in name:
            aux = block(aux)
        elif "aux_skip" in name:
            x = skips[-1] = x + block(aux)
        elif "aux_residual" in name:
            x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
        else:
            # x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
            x = block(x, emb) if block.__class__.__name__ == "UNetBlock" else block(x)
            skips.append(x)

    # Decoder.
    aux = None
    tmp = None
    for name, block in self.dec.items():
        if "aux_up" in name:
            aux = block(aux)
        elif "aux_norm" in name:
            tmp = block(x)
        elif "aux_conv" in name:
            tmp = block(silu(tmp))
            aux = tmp if aux is None else tmp + aux
        else:
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)

        # if return_feature_from == name:
        #     return x
        if name in return_feature_from:
            return_features.append(x)
            if len(return_features) == len(return_feature_from):
                return {"features": return_features, "emb": emb}
                # return (return_features, emb) if return_embed else (return_features, None)

    # return None if not return_embed else (None, emb)
    # return (None, None) if not return_embed else (None, emb)
    return {"features": None, "emb": None}


def sample_sigma(batch_size, device, P_mean=-1.2, P_std=1.2):
    # Sample sigma from the distribution
    rnd_normal = torch.randn([batch_size, 1, 1, 1], device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    return sigma


class PretrainedFeatureExtractor(nn.Module):
    # EDM featue extractor, default output is 256x16x16 (c, h, w)
    def __init__(
        self,
        network_pkl,
        device,
        feature_layer_names=[""],
        sigma=None,
        unconditional=False,
        perturb=False,
        cfg=False,
        freeze=True,
        num_classes=10,
        from_scratch=False,
        dual_time=False,
        duplicate_time=False,
    ):
        super(PretrainedFeatureExtractor, self).__init__()
        self.from_scratch = from_scratch
        # Load the pretrained model
        with dnnlib.util.open_url(network_pkl, verbose=True) as f:
            self.net = pickle.load(f)["ema"].to(device)
        if self.from_scratch:
            # to support diffusion backbone train from scratch
            from edm.training.networks import SongUNet
            new_model = SongUNet(img_resolution=32, in_channels=3, out_channels=3, channel_mult=[2,2,2]).to(device)

            # Compare the number of parameters of the new net and the pretrained net
            # num_trainable_params = sum(p.numel() for p in new_model.parameters())
            # num_pretrained_params = sum(p.numel() for p in self.net.model.parameters())
            # print(f"Number of trainable parameters in the new net: {num_trainable_params}")
            # print(f"Number of trainable parameters in the pretrained net: {num_pretrained_params}")
            # if num_trainable_params != num_pretrained_params:
            #     raise ValueError("The number of trainable parameters in the new net and the pretrained net are different.")

            self.net.model = new_model

        # Disable gradients (freeze the model)
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False
        else:
            # print out the number of trainable parameters in the model
            print("Warning: Not freezing the model parameters of diffusion feature extractor.")
            # unfreeze the parameters
            for param in self.net.parameters():
                param.requires_grad = True
            num_trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            print(f"Number of trainable parameters in the feature extractor: {num_trainable_params}")
            if num_trainable_params == 0:
                print("Warning: No trainable parameters in the feature extractor. Check if the model is frozen.")
            

        # Register the layer to extract features from
        # self.feature_layer_name = feature_layer_name
        # if feature_layer_name not in self.net.model.dec:
        #     raise ValueError(f"Layer {feature_layer_name} not found in the model.")
        for feature_layer_name in feature_layer_names:
            if feature_layer_name not in self.net.model.dec:
                raise ValueError(f"Layer {feature_layer_name} not found in the model.")
        # self.feature_layer = self.net.model.dec[feature_layer_name]
        self.net.model.forward = MethodType(new_model_forward, self.net.model)
        self.net.forward = MethodType(new_net_forward, self.net)
        # print(f"Feature layer name: {self.feature_layer_name}")
        self.feature_layer_names = feature_layer_names
        print(f"Feature layer names: {feature_layer_names}")
        self.sigma = sigma
        if self.sigma is None:
            print("Warning: Using sampled sigma value.")
        else:
            print(f"Using sigma: {self.sigma}")
        self.dual_time = dual_time
        self.duplicate_time = duplicate_time
        if self.dual_time:
            print(f"Warning: Using dual time step for diffusion alignment. One sampling the other with sigma {self.sigma}")
        if self.duplicate_time:
            print(f"Warning: Using duplicate time step for diffusion alignment. Both with sigma {self.sigma}")
        self.num_classes = num_classes
        self.unconditional = unconditional
        self.cfg = cfg
        if self.cfg:
            assert not unconditional, "Set unconditional to False when using CFG."
            print("Warning: Using CFG feature extractor.")
        if self.unconditional:
            print("Warning: Using unconditional feature extractor.")
        self.perturb = perturb
        if self.perturb:
            print("Warning: Using perturbed image as the input of the feature extractor.")
        self.fixed_perturbation_noise = None

    def get_perturb_image(self, x, sigma):
        if self.fixed_perturbation_noise is not None:
            noise = self.fixed_perturbation_noise.to(x.device).repeat(x.shape[0], 1, 1, 1) * sigma.view(-1, 1, 1, 1)
        else:
            noise = torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
        return x + noise

    def forward(self, x, class_labels=None, return_embed=False):
        # with torch.no_grad():
        bs = x.shape[0]
        x = x * 2.0 - 1.0  # Normalize input to [-1, 1]
        if self.sigma is None:
            sigma = sample_sigma(x.shape[0], x.device, P_mean=-1.2, P_std=1.2)
        else:
            sigma = torch.tensor(self.sigma).to(x.device).reshape(1, 1, 1, 1).repeat(x.shape[0], 1, 1, 1)
        

        if class_labels is not None and (class_labels.dim() == 1 or class_labels.size(1) == 1):
            d_type, d_device = class_labels.dtype, class_labels.device
            class_labels = torch.nn.functional.one_hot(class_labels, self.num_classes).to(
                device=d_device, dtype=d_type
            )
        if self.cfg:
            raise ValueError("CFG is disabled.")
            if class_labels is not None:
                class_null = torch.zeros_like(class_labels, device=class_labels.device, dtype=class_labels.dtype)
                class_labels = torch.cat([class_labels, class_null], dim=0)
            else:
                raise ValueError("class_labels should not be None when using CFG.")
            x = torch.cat([x, x], dim=0)
            sigma = torch.cat([sigma, sigma], dim=0)
        if self.dual_time:
            x = torch.cat([x, x], dim=0)
            if class_labels is not None:
                class_labels = torch.cat([class_labels, class_labels], dim=0)
            dual_sigma = sample_sigma(x.shape[0] // 2, x.device, P_mean=-1.2, P_std=1.2)
            sigma = torch.cat([sigma, dual_sigma], dim=0)
        elif self.duplicate_time:
            x = torch.cat([x, x], dim=0)
            if class_labels is not None:
                class_labels = torch.cat([class_labels, class_labels], dim=0)
            sigma = torch.cat([sigma, sigma], dim=0)

        if self.perturb:
            x = self.get_perturb_image(x, sigma)

        if self.unconditional:
            class_labels = None

        # Forward pass through the model and capture the feature at the specified layer

        # def hook_fn(_, __, output):
        #     nonlocal feature
        #     feature = output.detach()
            # raise RuntimeError("early")

        # sigma = torch.tensor(sigma).to(x.device).repeat(x.shape[0])
        # sigma = torch.tensor(self.sigma).to(x.device).repeat(x.shape[0])

        # Register hook to capture the feature during forward pass
        # hook = self.feature_layer.register_forward_hook(hook_fn)

        # Perform the forward pass

        result = self.net(
            x,
            sigma=sigma,
            class_labels=class_labels,
            # return_feature_from=self.feature_layer_name,
            return_feature_from=self.feature_layer_names,
            return_embed=return_embed,
        )
        sigma_data = 0.5
        # should only use weight in diffusion mse training rescaling
        # weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
        # result["weight"] = weight

        if self.cfg:
            raise NotImplementedError("CFG feature extraction is not implemented yet.")
            # feature_cond, feature_uncond = feature.chunk(2, dim=0)
            # return feature_cond, feature_uncond

        return result

    @property
    def cond_embed_dim(self):
        # map_layer1 out_features is the final embedding width
        return self.net.model.map_layer1.out_features

    # fix the perturbation noise instead of sampling from N(0, 1)
    def set_perturbation_noise(self, img_size):
        self.fixed_perturbation_noise = torch.randn(img_size)

    def unset_perturbation_noise(self):
        self.fixed_perturbation_noise = None



# Example usage
def get_feature_model(
    network_pkl="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl",
    device=torch.device("cuda"),
    feature_layer_name=["8x8_block4"],
    sigma=None,
    unconditional=False,
    perturb=False,
    cfg=False,
    dataset="cifar10",
    dual_time=False,
    duplicate_time=False,
):
    assert not (duplicate_time and dual_time), "Duplicate time and dual time cannot be used together."
    num_classes = 10
    if unconditional:
        print("Using unconditional feature extractor checkpoint.")
        network_pkl = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl"
    if "cifar100" in dataset:
        print("Using CIFAR-100 feature extractor checkpoint.")
        network_pkl = "https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/edm-cifar100-32x32-cond-vp.pkl"
        num_classes = 100
        if unconditional:
            raise NotImplementedError("Unconditional feature extractor for CIFAR-100 is not implemented yet.")
    elif "svhn" in dataset:
        print("Using SVHN feature extractor checkpoint.")
        network_pkl = "https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/edm-svhn-32x32-cond-vp.pkl"
        num_classes = 10
        if unconditional:
            raise NotImplementedError("Unconditional feature extractor for SVHN is not implemented yet.")
    elif "tiny-imagenet" in dataset:
        raise NotImplementedError("TinyImageNet feature extractor is not implemented yet.")
    model = PretrainedFeatureExtractor(network_pkl, device, feature_layer_name, sigma, unconditional, perturb, cfg, num_classes=num_classes, dual_time=dual_time, duplicate_time=duplicate_time)
    return model


if __name__ == "__main__":
    model = get_feature_model(perturb=True, cfg=True)
    print(model)
    bs = 1024
    print(f"{bs=}")
    input_tensor = torch.randn(bs, 3, 32, 32).cuda()  # Adjust size to your input
    class_labels = torch.randint(0, 10, (bs,)).cuda()  # Adjust size to your input
    class_labels = torch.nn.functional.one_hot(class_labels, num_classes=10)
    features = None
    for _ in tqdm(range(10)):
        features = model(input_tensor, class_labels)

    if features is not None:
        if isinstance(features, tuple):
            print(features[0].shape, features[1].shape)
        else:
            print(features.shape)
    else:
        raise ValueError("Feature extraction failed.")
