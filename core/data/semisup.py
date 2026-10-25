import os
import pickle
import numpy as np

import torch
import torch.distributed as dist


def get_semisup_dataloaders(train_dataset, test_dataset, val_dataset=None, batch_size=256, batch_size_test=256, num_workers=4, 
                            unsup_fraction=0.5, distributed=False, pin_memory=False):
    """
    Return dataloaders with custom sampling of pseudo-labeled data.
    """
    dataset_size = train_dataset.dataset_size
    train_batch_sampler = SemiSupervisedSampler(train_dataset.sup_indices, train_dataset.unsup_indices, batch_size, 
                                                unsup_fraction, num_batches=int(np.ceil(dataset_size/batch_size)), distributed=distributed)
    epoch_size = len(train_batch_sampler) * batch_size

    # kwargs = {'num_workers': num_workers, 'pin_memory': torch.cuda.is_available() }
    # kwargs = {'num_workers': num_workers, 'pin_memory': True}    
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    
    test_sampler = None
    val_sampler = None
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        if val_dataset:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, sampler=test_sampler, **kwargs)
    
    if val_dataset:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, sampler=val_sampler, **kwargs)
        return train_dataloader, test_dataloader, val_dataloader
    return train_dataloader, test_dataloader


class SemiSupervisedDataset(torch.utils.data.Dataset):
    """
    A dataset with auxiliary pseudo-labeled data.
    """
    def __init__(self, base_dataset='cifar10', take_amount=None, take_amount_seed=13, aux_data_filename=None, 
                 add_aux_labels=False, aux_take_amount=None, train=False, validation=False, **kwargs):

        self.base_dataset = base_dataset
        self.load_base_dataset(train, **kwargs)


        if validation:
            self.dataset.data = self.dataset.data[1024:]
            self.dataset.targets = self.dataset.targets[1024:]
        
        self.train = train

        if self.train:
            if take_amount is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices), take_amount, replace=False)
                np.random.set_state(rng_state)

                self.targets = self.targets[take_inds]
                self.data = self.data[take_inds]

            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            if aux_data_filename is not None:
                aux_path = aux_data_filename
                print('Loading data from %s' % aux_path)
                if os.path.splitext(aux_path)[1] == '.pickle':
                    # for data from Carmon et al, 2019.
                    with open(aux_path, 'rb') as f:
                        aux = pickle.load(f)
                    aux_data = aux['data']
                    aux_targets = aux['extrapolated_targets']
                else:
                    # for data from Rebuffi et al, 2021.
                    aux = np.load(aux_path)
                    aux_data = aux['image']
                    print(aux_data.shape)
                    aux_targets = aux['label']
                
                orig_len = len(self.data)

                if aux_take_amount is not None:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(aux_data), aux_take_amount, replace=False)
                    np.random.set_state(rng_state)

                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]

                self.data = np.concatenate((self.data, aux_data), axis=0)

                if not add_aux_labels:
                    self.targets.extend([-1] * len(aux_data))
                else:
                    self.targets.extend(aux_targets)
                self.unsup_indices.extend(range(orig_len, orig_len+len(aux_data)))

        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []
    
    def load_base_dataset(self, **kwargs):
        raise NotImplementedError()
    
    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def attach_latent_cache(self, latent_dir, flip=True):
        """
        Attach pre-computed VAE latents (teacher inputs for latent-based REPA).
        Expects latents.npy / latents_flip.npy / latents_stats.pt in latent_dir,
        index-aligned with this dataset (base train order, then aux order).
        Memmaps are opened lazily per worker process.
        """
        stats_path = os.path.join(latent_dir, "latents_stats.pt")
        if os.path.exists(stats_path):
            stats = torch.load(stats_path)
            self._latent_mean = stats["mean"].numpy().astype(np.float32)
            self._latent_std = stats["std"].numpy().astype(np.float32)
        else:
            # e.g. cached pooled feature targets: cosine loss is scale-invariant
            self._latent_mean = self._latent_std = None
        self.latent_dir = latent_dir
        self.latent_flip = flip
        self._latent_mm = None
        self._latent_mm_flip = None
        n = np.lib.format.open_memmap(os.path.join(latent_dir, "latents.npy"), mode="r").shape[0]
        # The cache is indexed over the FULL base train set followed by the aux set.
        # With validation=True the dataset drops the first 1024 base images, so map
        # dataset indices to cache indices via the base offset.
        self._latent_n_sup = len(self.sup_indices)
        self._latent_aux_start = n - len(self.unsup_indices)
        self._latent_base_offset = self._latent_aux_start - self._latent_n_sup
        assert self._latent_base_offset >= 0, (
            f"latent cache base ({self._latent_aux_start}) smaller than dataset base ({self._latent_n_sup})"
        )

    def _get_latent(self, item):
        if self._latent_mm is None:
            self._latent_mm = np.lib.format.open_memmap(os.path.join(self.latent_dir, "latents.npy"), mode="r")
            self._latent_mm_flip = np.lib.format.open_memmap(os.path.join(self.latent_dir, "latents_flip.npy"), mode="r")
        if item < self._latent_n_sup:
            cache_idx = item + self._latent_base_offset
        else:
            cache_idx = self._latent_aux_start + (item - self._latent_n_sup)
        # Random flip choice mirrors the in1k PairedImageLatentDataset protocol.
        mm = self._latent_mm_flip if (self.latent_flip and np.random.uniform() > 0.5) else self._latent_mm
        z = np.asarray(mm[cache_idx], dtype=np.float32)
        if self._latent_mean is not None:
            z = (z - self._latent_mean[0]) / self._latent_std[0]
        return torch.from_numpy(z)

    def __getitem__(self, item):
        self.dataset.labels = self.targets
        if getattr(self, "latent_dir", None) is not None:
            x, y = self.dataset[item]
            return x, self._get_latent(item), y
        return self.dataset[item]


class SemiSupervisedDatasetSVHN(torch.utils.data.Dataset):
    """
    A dataset with auxiliary pseudo-labeled data.
    """
    def __init__(self, base_dataset='svhn', take_amount=None, take_amount_seed=13, aux_data_filename=None, 
                 add_aux_labels=False, aux_take_amount=None, train=False, validation=False, **kwargs):

        self.base_dataset = base_dataset
        self.load_base_dataset(train, **kwargs)
        self.dataset.labels = self.dataset.labels.tolist()


        if validation:
            self.dataset.data = self.dataset.data[1024:]
            self.dataset.labels = self.dataset.labels[1024:]
        
        self.train = train

        if self.train:
            if take_amount is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices), take_amount, replace=False)
                np.random.set_state(rng_state)

                self.targets = self.targets[take_inds]
                self.data = self.data[take_inds]

            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            if aux_data_filename is not None:
                aux_path = aux_data_filename
                print('Loading data from %s' % aux_path)
                if os.path.splitext(aux_path)[1] == '.pickle':
                    # for data from Carmon et al, 2019.
                    with open(aux_path, 'rb') as f:
                        aux = pickle.load(f)
                    aux_data = aux['data']
                    aux_targets = aux['extrapolated_targets']
                else:
                    # for data from Rebuffi et al, 2021.
                    aux = np.load(aux_path)
                    aux_data = aux['image']
                    print(aux_data.shape)
                    aux_targets = aux['label']

                orig_len = len(self.data)

                if aux_take_amount is not None:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(aux_data), aux_take_amount, replace=False)
                    np.random.set_state(rng_state)

                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]

                self.data = np.concatenate((self.data, aux_data.transpose(0,3,1,2)), axis=0)

                if not add_aux_labels:
                    self.targets.extend([-1] * len(aux_data))
                else:
                    self.targets.extend(aux_targets)
                self.unsup_indices.extend(range(orig_len, orig_len+len(aux_data)))

        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []
    
    def load_base_dataset(self, **kwargs):
        raise NotImplementedError()
    
    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.labels

    @targets.setter
    def targets(self, value):
        self.dataset.labels = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets
        return self.dataset[item]
    
    
class SemiSupervisedSampler(torch.utils.data.Sampler):
    """
    Balanced sampling from the labeled and unlabeled data.
    """
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5, num_batches=None, distributed=False):
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size

        if distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            # Partition indices based on rank
            # Using striding to partition
            self.sup_inds = self.sup_inds[rank::world_size]
            self.unsup_inds = self.unsup_inds[rank::world_size]

        if num_batches is not None:
            self.num_batches = num_batches
            if distributed:
                # If num_batches was provided (usually based on full dataset), scale it down
                # We assume num_batches passed in was calculated for the full dataset
                # If the caller calculated it based on partitioned dataset, this would be double scaling.
                # But in get_semisup_dataloaders, it calculates:
                # num_batches=int(np.ceil(dataset_size/batch_size))
                # dataset_size is full.
                # So we should scale it.
                # Wait, get_semisup_dataloaders calculates num_batches using dataset_size.
                # dataset_size is from train_dataset.dataset_size which is full size.
                # So yes, we need to scale it.
                self.num_batches = int(np.ceil(self.num_batches / dist.get_world_size()))
        else:
            self.num_batches = int(np.ceil(len(self.sup_inds) / self.sup_batch_size))
        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i]
                                 for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in torch.randint(high=len(self.unsup_inds), 
                                                                            size=(self.batch_size - len(batch),), 
                                                                            dtype=torch.int64)])
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches
