import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import argparse
import os
from safetensors.torch import save_file
from datetime import datetime
from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer.vavae import VA_VAE

class ImageFolderWithIndex(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample, target, index

def main(args):
    """
    Run a tokenizer on full dataset and save the features.
    """
    assert torch.cuda.is_available(), "Extract features currently requires at least one GPU."

    # Setup DDP:
    try:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        world_size = dist.get_world_size()
        seed = args.seed + rank
        if rank == 0:
            print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    except:
        print("Failed to initialize DDP. Running in local mode.")
        rank = 0
        device = 0
        world_size = 1
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup feature folders:
    output_dir = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.config))[0], f'{args.data_split}_{args.image_size}')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # Create model:
    tokenizer = VA_VAE(
        args.config
    )

    # Setup data:
    datasets = [
        ImageFolderWithIndex(args.data_path, transform=tokenizer.img_transform(p_hflip=0.0)),
        ImageFolderWithIndex(args.data_path, transform=tokenizer.img_transform(p_hflip=1.0))
    ]
    samplers = [
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.seed
        ) for dataset in datasets
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        ) for dataset, sampler in zip(datasets, samplers)
    ]
    total_data_in_loop = len(loaders[0].dataset)
    if rank == 0:
        print(f"Total data in one loop: {total_data_in_loop}")

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    indices = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images} of {total_data_in_loop} images')
        
        for loader_idx, data in enumerate(batch_data):
            x = data[0]
            y = data[1]  # (N,)
            idx = data[2] # 4. Get indices from loader
            z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)

            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            
            if loader_idx == 0:
                latents.append(z)
                labels.append(y)
                indices.append(idx) # 5. Store indices
            else:
                latents_flip.append(z)

        if len(latents) == args.shard_size // args.batch_size:
            latents = torch.cat(latents, dim=0)
            latents_flip = torch.cat(latents_flip, dim=0)
            labels = torch.cat(labels, dim=0)
            indices = torch.cat(indices, dim=0) # 6. Concatenate indices
            # Convert to target dtype for saving
            dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
            save_dtype = dtype_map[args.save_dtype]
            save_dict = {
                'latents': latents.to(save_dtype),
                'latents_flip': latents_flip.to(save_dtype),
                'labels': labels,
                'indices': indices
            }
            for key in save_dict:
                if rank == 0:
                    print(key, save_dict[key].shape)
            save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
            )
            if rank == 0:
                print(f'Saved {save_filename}')
            
            latents = []
            latents_flip = []
            labels = []
            indices = [] # 7. Reset indices
            saved_files += 1

    # save remainder latents
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        indices = torch.cat(indices, dim=0) # 8. Handle remainder
        # Convert to target dtype for saving
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        save_dtype = dtype_map[args.save_dtype]
        save_dict = {
            'latents': latents.to(save_dtype),
            'latents_flip': latents_flip.to(save_dtype),
            'labels': labels,
            'indices': indices
        }
        for key in save_dict:
            if rank == 0:
                print(key, save_dict[key].shape)
        save_filename = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
        )
        if rank == 0:
            print(f'Saved {save_filename}')

    # Calculate latents stats
    dist.barrier()
    if rank == 0:
        dataset = ImgLatentDataset(output_dir, latent_norm=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/path/to/your/data')
    parser.add_argument("--data_split", type=str, default='imagenet_train')
    parser.add_argument("--output_path", type=str, default="/path/to/your/output")
    parser.add_argument("--config", type=str, default="config_details.yaml")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of images per safetensors shard")
    parser.add_argument("--save_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Dtype for saving latents")
    args = parser.parse_args()
    main(args)