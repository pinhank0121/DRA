import os
import torch
import torch.distributed as dist

from .cifar10 import load_cifar10
from .cifar100 import load_cifar100
from .svhn import load_svhn
from .cifar10s import load_cifar10s
from .cifar100s import load_cifar100s
from .svhns import load_svhns
from .tiny_imagenet import load_tinyimagenet
from .tiny_imagenets import load_tinyimagenets
from .imagenet import load_imagenet

from .semisup import get_semisup_dataloaders


SEMISUP_DATASETS = ['cifar10s', 'cifar100s', 'svhns', 'tiny-imagenets']
DATASETS = ['cifar10', 'svhn', 'cifar100', 'tiny-imagenet', 'imagenet'] + SEMISUP_DATASETS

_LOAD_DATASET_FN = {
    'cifar10': load_cifar10,
    'cifar100': load_cifar100,
    'svhn': load_svhn,
    'tiny-imagenet': load_tinyimagenet,
    'cifar10s': load_cifar10s,
    'cifar100s': load_cifar100s,
    'svhns': load_svhns,
    'tiny-imagenets': load_tinyimagenets,
    'imagenet': load_imagenet,
}


def get_data_info(data_dir):
    """
    Returns dataset information.
    Arguments:
        data_dir (str): path to data directory.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    if 'cifar100' in data_dir:
        from .cifar100 import DATA_DESC
    elif 'cifar10' in data_dir:
        from .cifar10 import DATA_DESC
    elif 'svhn' in data_dir:
        from .svhn import DATA_DESC
    elif 'tiny-imagenet' in data_dir:
        from .tiny_imagenet import DATA_DESC
    elif 'imagenet' in data_dir:
        from .imagenet import DATA_DESC
    else:
        raise ValueError(f'Only data in {DATASETS} are supported!')
    DATA_DESC['data'] = dataset
    return DATA_DESC


def load_data(data_dir, batch_size=256, batch_size_test=256, num_workers=4, use_augmentation='base', use_consistency=False, shuffle_train=True, 
              aux_data_filename=None, unsup_fraction=None, validation=False, distributed=False):
    """
    Returns train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        batch_size (int): batch size for training.
        batch_size_test (int): batch size for validation.
        num_workers (int): number of workers for loading the data.
        use_augmentation (base/none): whether to use augmentations for training set.
        shuffle_train (bool): whether to shuffle training set.
        aux_data_filename (str): path to unlabelled data.
        unsup_fraction (float): fraction of unlabelled data per batch.
        validation (bool): if True, also returns a validation dataloader for unspervised cifar10 (as in Gowal et al, 2020).
        distributed (bool): whether to use DistributedDataParallel.
    """
    dataset = os.path.basename(os.path.normpath(data_dir))
    load_dataset_fn = _LOAD_DATASET_FN[dataset]
    pin_memory = "imagenet" in dataset
    
    if validation:
        assert dataset in SEMISUP_DATASETS, 'Only semi-supervised datasets allow a validation set.'
        train_dataset, test_dataset, val_dataset = load_dataset_fn(data_dir=data_dir, use_augmentation=use_augmentation, use_consistency=use_consistency,
                                                                   aux_data_filename=aux_data_filename, validation=True)
    else:
        if dataset in SEMISUP_DATASETS or (dataset == 'imagenet' and aux_data_filename):
            _ret = load_dataset_fn(data_dir=data_dir, use_augmentation=use_augmentation, aux_data_filename=aux_data_filename)
            # some loaders (e.g. cifar100s) always return (train, test, val=None)
            train_dataset, test_dataset = _ret[0], _ret[1]
        else:
            train_dataset, test_dataset = load_dataset_fn(data_dir=data_dir, use_augmentation=use_augmentation)
       
    if dataset in SEMISUP_DATASETS or (dataset == 'imagenet' and aux_data_filename):
        if validation:
            train_dataloader, test_dataloader, val_dataloader = get_semisup_dataloaders(
                train_dataset, test_dataset, val_dataset, batch_size, batch_size_test, num_workers, unsup_fraction, distributed=distributed, pin_memory=pin_memory
            )
        else:
            train_dataloader, test_dataloader = get_semisup_dataloaders(
                train_dataset, test_dataset, None, batch_size, batch_size_test, num_workers, unsup_fraction, distributed=distributed, pin_memory=pin_memory
            )
    else:
        
        train_sampler = None
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            shuffle_train = False
            
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, 
                                                       num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)
        
        test_sampler = None
        if distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, 
                                                      num_workers=num_workers, pin_memory=pin_memory, sampler=test_sampler)
                                                      
    if validation:
        return train_dataset, test_dataset, val_dataset, train_dataloader, test_dataloader, val_dataloader
    return train_dataset, test_dataset, train_dataloader, test_dataloader


def load_paired_data(
    data_dir, 
    latent_dir, 
    batch_size=256, 
    batch_size_test=256, 
    num_workers=4, 
    use_augmentation='base',
    distributed=False,
    aux_data_filename=None,
    aux_latent_dir=None,
    unsup_fraction=None,
):
    """
    Load paired (image, latent, label) dataset for REPA training with LightningDiT.
    
    Supports optional aux/synthetic data with its own latent directory.
    Uses semi-supervised sampling when aux data is provided.
    
    Args:
        data_dir: Path to ImageNet data directory (containing 'train' and 'val' subdirs)
        latent_dir: Path to pre-computed VAE latents directory for ImageNet
        batch_size: Batch size for training
        batch_size_test: Batch size for testing
        num_workers: Number of data loading workers
        use_augmentation: Augmentation type ('base' or 'none')
        distributed: Whether to use DistributedDataParallel
        aux_data_filename: Path to aux/synthetic data directory (optional)
        aux_latent_dir: Path to pre-computed VAE latents for aux data (required if aux_data_filename is set)
        unsup_fraction: Fraction of unsupervised (aux) data per batch (used with get_semisup_dataloaders)
    
    Returns:
        train_dataset, test_dataset, train_dataloader, test_dataloader
    """
    from .imagenet import load_paired_imagenet
    
    # Load paired datasets
    train_dataset, test_dataset = load_paired_imagenet(
        data_dir=data_dir,
        latent_dir=latent_dir,
        use_augmentation=use_augmentation,
        aux_data_filename=aux_data_filename,
        aux_latent_dir=aux_latent_dir,
    )
    
    # Use semi-supervised dataloaders if aux data is provided (has sup_indices/unsup_indices)
    if hasattr(train_dataset, 'sup_indices') and hasattr(train_dataset, 'unsup_indices'):
        train_dataloader, test_dataloader = get_semisup_dataloaders(
            train_dataset, 
            test_dataset, 
            None,  # No validation set
            batch_size, 
            batch_size_test, 
            num_workers, 
            unsup_fraction,
            distributed=distributed, 
            pin_memory=True
        )
    else:
        # Standard data loaders for ImageNet-only (no aux data)
        train_sampler = None
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            sampler=train_sampler,
        )
        
        test_sampler = None
        if distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size_test,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            sampler=test_sampler,
        )
    
    return train_dataset, test_dataset, train_dataloader, test_dataloader
