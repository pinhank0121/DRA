import torch

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
# from torchvision.datasets import ImageNet
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import ConcatDataset
import os


DATA_DESC = {
    'data': 'imagenet',
    'classes': tuple(range(0, 1000)),
    'num_classes': 1000,
    'mean': [0.485, 0.456, 0.406], 
    'std': [0.229, 0.224, 0.225],
}


def get_imagenet_transforms(use_augmentation='base'):
    """
    Returns train and test transforms for ImageNet.
    """
    if isinstance(use_augmentation, str) and '256' in use_augmentation:
        crop_size = 256
    else:
        crop_size = 224
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
    ])
    if use_augmentation == 'base':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    elif use_augmentation == 'rrc':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    elif use_augmentation == 'base-256':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    elif use_augmentation == "base_randaugment":
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    elif use_augmentation == 'rrc_randaugment':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
            transforms.RandAugment(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    else: 
        train_transform = test_transform
    return train_transform, test_transform


def load_imagenet(data_dir, use_augmentation='base', aux_data_filename=None, **kwargs):
    """
    Returns ImageNet train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
        aux_data_filename (str): path to unlabelled data (optional).
    Returns:
        train dataset, test dataset. 
    """
    train_transform, test_transform = get_imagenet_transforms(use_augmentation)
    
    train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=test_transform)

    if aux_data_filename and os.path.isdir(aux_data_filename):
        print(f"Loading ImageNet + Synthetic data from {aux_data_filename}")
        syn_dataset = ImageFolder(root=aux_data_filename, transform=train_transform)
        
        sup_indices = list(range(len(train_dataset)))
        unsup_indices = list(range(len(train_dataset), len(train_dataset) + len(syn_dataset)))
        
        # Store the original size of the supervised dataset for epoch length calculation
        dataset_size = len(train_dataset)
        
        train_dataset = ConcatDataset([train_dataset, syn_dataset])
        train_dataset.sup_indices = sup_indices
        train_dataset.unsup_indices = unsup_indices
        train_dataset.dataset_size = dataset_size

    return train_dataset, test_dataset


def load_paired_imagenet(
    data_dir, 
    latent_dir, 
    use_augmentation='base', 
    aux_data_filename=None, 
    aux_latent_dir=None,
    **kwargs
):
    """
    Returns paired (image, latent, label) ImageNet train and test datasets.
    
    This is for REPA training with LightningDiT where we need both raw images 
    (for classifier) and pre-computed VAE latents (for diffusion feature extraction).
    
    Arguments:
        data_dir (str): path to ImageNet data directory (with 'train' and 'val' subdirs).
        latent_dir (str): path to pre-computed VAE latents for ImageNet.
        use_augmentation (base/none): whether to use augmentations for training set.
        aux_data_filename (str): path to aux/synthetic image data (optional).
        aux_latent_dir (str): path to pre-computed VAE latents for aux data (required if aux_data_filename is set).
    
    Returns:
        train_dataset, test_dataset
    """
    from .img_latent_pair import PairedImageLatentDataset
    
    train_transform, test_transform = get_imagenet_transforms(use_augmentation)
    
    # Load main ImageNet paired dataset
    imagenet_dataset = PairedImageLatentDataset(
        latent_dir=latent_dir,
        image_dir=os.path.join(data_dir, 'train'),
        image_transform=train_transform,
        latent_norm=True,
        flip_latents=True,
    )
    
    # Test dataset (standard ImageNet, no latents needed for evaluation)
    test_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=test_transform)
    
    # Handle aux/synthetic data if provided
    if aux_data_filename and os.path.isdir(aux_data_filename):
        if aux_latent_dir is None:
            raise ValueError("aux_latent_dir is required when using aux_data_filename with paired ImageNet.")
        
        print(f"Loading Paired ImageNet + Synthetic data:")
        print(f"  - ImageNet images: {os.path.join(data_dir, 'train')}")
        print(f"  - ImageNet latents: {latent_dir}")
        print(f"  - Aux images: {aux_data_filename}")
        print(f"  - Aux latents: {aux_latent_dir}")
        
        # Load aux/synthetic paired dataset
        aux_dataset = PairedImageLatentDataset(
            latent_dir=aux_latent_dir,
            image_dir=aux_data_filename,
            image_transform=train_transform,
            latent_norm=True,
            flip_latents=True,
        )
        
        # Store indices for semi-supervised sampling
        sup_indices = list(range(len(imagenet_dataset)))
        unsup_indices = list(range(len(imagenet_dataset), len(imagenet_dataset) + len(aux_dataset)))
        dataset_size = len(imagenet_dataset)  # Original supervised dataset size
        
        # Concatenate datasets
        train_dataset = ConcatDataset([imagenet_dataset, aux_dataset])
        train_dataset.sup_indices = sup_indices
        train_dataset.unsup_indices = unsup_indices
        train_dataset.dataset_size = dataset_size
        
        print(f"  - ImageNet samples: {len(imagenet_dataset)}")
        print(f"  - Aux samples: {len(aux_dataset)}")
        print(f"  - Combined samples: {len(train_dataset)}")
    else:
        train_dataset = imagenet_dataset
        print(f"Loading Paired ImageNet data:")
        print(f"  - ImageNet images: {os.path.join(data_dir, 'train')}")
        print(f"  - ImageNet latents: {latent_dir}")
        print(f"  - Total samples: {len(train_dataset)}")
    
    return train_dataset, test_dataset
