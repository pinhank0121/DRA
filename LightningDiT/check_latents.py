import torch
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import ImageFolder
from datasets.img_latent_dataset import ImgLatentDataset
from safetensors import safe_open
from tokenizer.vavae import VA_VAE
from torchvision.utils import save_image
from scipy.stats import norm

# def sample_logit_normal(mu=0, sigma=1, size=1):
#     # Generate samples from the normal distribution
#     samples = norm.rvs(loc=mu, scale=sigma, size=size)
    
#     # Transform samples to be in the range (0, 1) using the logistic function
#     # Scale to 0.5 to 1.0
#     samples = 1 / (1 + np.exp(-samples)) * 0.5 + 0.5

#     # Numpy to Tensor
#     samples = torch.tensor(samples, dtype=torch.float32)

#     return samples

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

def add_noise_to_latent(latent, time_step=0.05):
    noise = torch.randn_like(latent)
    return (1 - time_step) * noise + time_step * latent

def add_text_to_image(img_tensor, text, font_size=14):
    """
    Add text label to the top-left corner of an image tensor.
    img_tensor: (C, H, W) tensor in range [0, 1]
    Returns: (C, H, W) tensor with text overlay
    """
    # Convert tensor to PIL Image
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np).convert("RGBA")
    
    # Create a transparent overlay for the text background
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Try to use a basic font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((4, 4), text, font=font)
    padding = 4
    rect_bbox = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)
    
    # Draw semi-transparent background
    draw.rectangle(rect_bbox, fill=(0, 0, 0, 180))
    
    # Draw text in white
    draw.text((4, 4), text, fill=(255, 255, 255, 255), font=font)
    
    # Composite overlay onto image
    img_pil = Image.alpha_composite(img_pil, overlay).convert("RGB")
    
    # Convert back to tensor
    img_tensor_out = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1) / 255.0
    
    return img_tensor_out

def decode_latent(vae, latent, dataset):
    """Helper function to decode a latent and return a visualization tensor."""
    z_rec = latent.unsqueeze(0).clone()
    if dataset.latent_norm:
        z_rec = z_rec / dataset.latent_multiplier
        z_rec = z_rec * dataset._latent_std + dataset._latent_mean
    
    device = dataset._latent_mean.device if hasattr(dataset, '_latent_mean') else 'cuda'
    decoded_np = vae.decode_to_images(z_rec.to(device))[0]
    decoded_vis = torch.from_numpy(decoded_np).float().permute(2, 0, 1) / 255.0
    return decoded_vis

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
            tensor_key = "latents" 
            latent = f.get_slice(tensor_key)[img_idx:img_idx+1]
            
            # Load the original index of this image in the ImageFolder
            original_idx_tensor = f.get_slice('indices')[img_idx:img_idx+1]
            original_idx = original_idx_tensor.item()

        # 3. Normalization (from parent class)
        if self.latent_norm:
            latent = (latent - self._latent_mean) / self._latent_std
        latent = latent * self.latent_multiplier
        latent = latent.squeeze(0)

        # 4. Load the corresponding raw image using the retrieved index
        # ImageFolder.__getitem__ returns (image, label)
        raw_image, _ = self.image_folder[original_idx]

        return raw_image, latent


def main(args):
    # 1. Setup VAE for decoding
    print(f"Loading VAE from {args.config}")
    vae = VA_VAE(args.config)
    
    # 2. Setup Dataset
    # We use the VAE's transform for the raw images so they match what the VAE expects
    # (center crop, normalize, etc.)
    transform = vae.img_transform(p_hflip=0.0) 
    
    dataset = PairedImageLatentDataset(
        latent_dir=args.latent_path,
        image_dir=args.data_path,
        image_transform=transform,
        latent_norm=True  # Important if your training uses normalization
    )
    
    print(f"Dataset size: {len(dataset)}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fixed time steps for Row 1: 0.1 to 1.0
    # fixed_time_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fixed_time_steps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # Row 1 will have: Original + Decoded + 10 noised = 12 images
    # Row 2 will have: 12 logit-normal sampled images to match
    num_logit_samples = 12
    
    # 3. Check a few samples
    indices = np.random.choice(len(dataset), args.num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        raw_img, latent = dataset[idx]
        
        # Raw image is currently normalized to [-1, 1], convert back to [0, 1] for visualization
        raw_vis = ((raw_img + 1) / 2).clamp(0, 1)
        
        # ========== ROW 1: Original, Decoded, Fixed time steps 0.1 to 1.0 ==========
        row1_images = []
        
        # Original image
        row1_images.append(add_text_to_image(raw_vis, "Original"))
        
        # Decoded image (no noise, t=1.0 means pure latent)
        decoded_vis = decode_latent(vae, latent, dataset)
        row1_images.append(add_text_to_image(decoded_vis, "Decoded"))
        
        # Noised images at fixed time steps
        for t in fixed_time_steps:
            noised_latent = add_noise_to_latent(latent, time_step=t)
            noised_vis = decode_latent(vae, noised_latent, dataset)
            row1_images.append(add_text_to_image(noised_vis, f"t={t:.1f}"))
        
        # ========== ROW 2: Logit-normal sampled time steps ==========
        row2_images = []
        logit_samples = sample_timestep_lognorm_like(batch_size=num_logit_samples, device=latent.device, low=0.7, high=0.9, sigma=1.0)
        
        for j, t_tensor in enumerate(logit_samples):
            t_val = t_tensor.item()
            noised_latent = add_noise_to_latent(latent, time_step=t_val)
            noised_vis = decode_latent(vae, noised_latent, dataset)
            row2_images.append(add_text_to_image(noised_vis, f"logit_t={t_val:.3f}"))
        
        # ========== Combine rows into grid ==========
        row1 = torch.cat(row1_images, dim=2)  # Concatenate along width
        row2 = torch.cat(row2_images, dim=2)  # Concatenate along width
        combined = torch.cat([row1, row2], dim=1)  # Concatenate along height (stack rows)
        
        save_path = os.path.join(args.output_dir, f"check_{i:03d}_idx{idx}.png")
        save_image(combined, save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to original ImageNet train folder")
    parser.add_argument("--latent_path", type=str, required=True, help="Path to folder containing .safetensors")
    parser.add_argument("--config", type=str, default="config_details.yaml", help="Path to VAE config")
    parser.add_argument("--output_dir", type=str, default="sanity_check_images")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    
    main(args)