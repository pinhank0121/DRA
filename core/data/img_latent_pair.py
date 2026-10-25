import os
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from safetensors import safe_open
from torchvision.datasets import ImageFolder

class ImgLatentDataset(Dataset):
    def __init__(self, data_dir, latent_norm=True, latent_multiplier=1.0, flip_latents=True):
        self.data_dir = data_dir
        self.latent_norm = latent_norm
        self.latent_multiplier = latent_multiplier
        self.flip_latents = flip_latents

        self.files = sorted(glob(os.path.join(data_dir, "*.safetensors")))
        self.img_to_file_map = self.get_img_to_safefile_map()
        
        if latent_norm:
            self._latent_mean, self._latent_std = self.get_latent_stats()

    def get_img_to_safefile_map(self):
        img_to_file = {}
        for safe_file in self.files:
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                labels = f.get_slice('labels')
                labels_shape = labels.get_shape()
                num_imgs = labels_shape[0]
                cur_len = len(img_to_file)
                for i in range(num_imgs):
                    img_to_file[cur_len+i] = {
                        'safe_file': safe_file,
                        'idx_in_file': i
                    }
        return img_to_file

    def get_latent_stats(self):
        latent_stats_cache_file = os.path.join(self.data_dir, "latents_stats.pt")
        if not os.path.exists(latent_stats_cache_file):
            latent_stats = self.compute_latent_stats()
            torch.save(latent_stats, latent_stats_cache_file)
        else:
            latent_stats = torch.load(latent_stats_cache_file)
        return latent_stats['mean'], latent_stats['std']
    
    def compute_latent_stats(self):
        num_samples = min(10000, len(self.img_to_file_map))
        random_indices = np.random.choice(len(self.img_to_file_map), num_samples, replace=False)
        
        sum_latents = None
        sum_sq_latents = None
        total_pixels = 0

        for idx in tqdm(random_indices):
            img_info = self.img_to_file_map[idx]
            safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
            with safe_open(safe_file, framework="pt", device="cpu") as f:
                features = f.get_slice('latents')
                # Load as float64 to prevent precision loss during accumulation
                feature = features[img_idx:img_idx+1].to(dtype=torch.float64)
                
                if sum_latents is None:
                    # Initialize accumulators with shape (1, C, 1, 1)
                    sum_latents = torch.zeros_like(feature.sum(dim=[0, 2, 3], keepdim=True))
                    sum_sq_latents = torch.zeros_like(sum_latents)

                sum_latents += feature.sum(dim=[0, 2, 3], keepdim=True)
                sum_sq_latents += (feature ** 2).sum(dim=[0, 2, 3], keepdim=True)
                total_pixels += feature.shape[0] * feature.shape[2] * feature.shape[3]

        mean = sum_latents / total_pixels
        # Calculate variance using: Var = (Sum(x^2) - (Sum(x)^2 / N)) / (N - 1)
        var = (sum_sq_latents - (sum_latents ** 2) / total_pixels) / (total_pixels - 1)
        std = torch.sqrt(var)
        
        # Convert back to float32
        latent_stats = {'mean': mean.float(), 'std': std.float()}
        print(latent_stats)
        return latent_stats

    def __len__(self):
        return len(self.img_to_file_map.keys())

    def __getitem__(self, idx):
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            if self.flip_latents:
                tensor_key = "latents_flip" if (np.random.uniform(0, 1) > 0.5) else "latents"
            else:
                tensor_key = "latents"
            features = f.get_slice(tensor_key)
            labels = f.get_slice('labels')
            feature = features[img_idx:img_idx+1]
            label = labels[img_idx:img_idx+1]

        if self.latent_norm:
            feature = (feature - self._latent_mean) / self._latent_std
        feature = feature * self.latent_multiplier
        
        # remove the first batch dimension (=1) kept by get_slice()
        feature = feature.squeeze(0)
        label = label.squeeze(0)
        return feature, label

class PairedImageLatentDataset(ImgLatentDataset):
    """
    Loads (Image, Latent) pairs.
    Uses safetensors for latents/indices and ImageFolder for raw images.
    """
    def __init__(self, latent_dir, image_dir, image_transform=None, **kwargs):
        # Initialize the latent dataset logic (scans .safetensors)
        super().__init__(latent_dir, **kwargs)
        
        # Initialize the standard image folder to load raw images on demand
        self.image_folder = ImageFolder(image_dir, transform=image_transform)

    def __getitem__(self, idx):
        # 1. Look up where the latent is stored
        img_info = self.img_to_file_map[idx]
        safe_file, img_idx = img_info['safe_file'], img_info['idx_in_file']
        
        # 2. Open safetensor to get latent AND the original dataset index
        with safe_open(safe_file, framework="pt", device="cpu") as f:
            # Load latent (randomly flip if your training requires it, or fixed)
            # For sanity check, we usually check the non-flipped one or just 'latents'
            tensor_key = "latents" if np.random.uniform(0, 1) > 0.5 else "latents_flip"
            latent = f.get_slice(tensor_key)[img_idx:img_idx+1]
            
            # Load the original index of this image in the ImageFolder
            original_idx_tensor = f.get_slice('indices')[img_idx:img_idx+1]
            original_idx = original_idx_tensor.item()
            
            # Load the label
            label = f.get_slice('labels')[img_idx:img_idx+1].squeeze(0)

        # 3. Normalization (from parent class)
        if self.latent_norm:
            latent = (latent - self._latent_mean) / self._latent_std
        latent = latent * self.latent_multiplier
        latent = latent.squeeze(0)

        # 4. Load the corresponding raw image using the retrieved index
        # ImageFolder.__getitem__ returns (image, label)
        raw_image, _ = self.image_folder[original_idx]

        return raw_image, latent, label