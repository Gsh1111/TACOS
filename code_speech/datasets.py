import os
import random
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets.shd import SpikingSpeechCommands
from spikingjelly.datasets import pad_sequence_collate


# =========================================================
# Reproducibility helpers
# =========================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """
    Ensure numpy/random/torch in each worker are deterministically seeded.
    The base seed comes from DataLoader(generator=...).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def resolve_dataset_seed(config, dataset_seed=None) -> int:
    """
    Priority:
    1) explicit function argument
    2) config.dataset_seed if exists
    3) fallback to config.seed
    """
    if dataset_seed is not None:
        return int(dataset_seed)
    if hasattr(config, "dataset_seed"):
        return int(config.dataset_seed)
    return int(config.seed)


class TimeNeurons_mask_aug(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, x, y):
        # Time mask
        if np.random.uniform() < self.config.TN_mask_aug_proba:
            mask_size = int(self.config.time_mask_proportion * x.shape[0])
            if mask_size > 0 and x.shape[0] > mask_size:
                ind = np.random.randint(0, x.shape[0] - mask_size)
                x[ind:ind + mask_size, :] = 0

        # Neuron mask
        if np.random.uniform() < self.config.TN_mask_aug_proba:
            max_mask = int(self.config.neuron_mask_size)
            if max_mask > 0 and x.shape[1] > max_mask:
                mask_size = np.random.randint(0, max_mask)
                ind = np.random.randint(0, x.shape[1] - max_mask)
                x[:, ind:ind + mask_size] = 0

        return x, y


class Augs(object):
    def __init__(self, config):
        self.config = config
        self.augs = [TimeNeurons_mask_aug(config)]

    def __call__(self, x, y):
        for aug in self.augs:
            x, y = aug(x, y)
        return x, y

    def list_augs(self):
        return [aug.__class__.__name__ for aug in self.augs]


# =========================================================
# SHD / SSC dataloaders
# =========================================================
def SHD_dataloaders(config, dataset_seed=None):
    dataset_seed = resolve_dataset_seed(config, dataset_seed)

    if config.use_aug:
        augs = Augs(config)
        print("Data Augmentations used:", augs.list_augs())
        train_dataset = BinnedSpikingHeidelbergDigits(
            config.datasets_path,
            config.n_bins,
            train=True,
            data_type='frame',
            duration=config.duration,
            transform=augs
        )
    else:
        train_dataset = BinnedSpikingHeidelbergDigits(
            config.datasets_path,
            config.n_bins,
            train=True,
            data_type='frame',
            duration=config.duration,
            transform=None
        )

    test_dataset = BinnedSpikingHeidelbergDigits(
        config.datasets_path,
        config.n_bins,
        train=False,
        data_type='frame',
        duration=config.duration
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=pad_sequence_collate,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(dataset_seed + 0),
    )

    test_loader = DataLoader(
        test_dataset,
        collate_fn=pad_sequence_collate,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(dataset_seed + 1),
    )

    return train_loader, test_loader


def SSC_dataloaders(config, dataset_seed=None):
    dataset_seed = resolve_dataset_seed(config, dataset_seed)

    if config.use_aug:
        augs = Augs(config)
        print("Data Augmentations used:", augs.list_augs())
        train_dataset = BinnedSpikingSpeechCommands(
            config.datasets_path,
            config.n_bins,
            split='train',
            data_type='frame',
            duration=config.duration,
            transform=augs
        )
    else:
        train_dataset = BinnedSpikingSpeechCommands(
            config.datasets_path,
            config.n_bins,
            split='train',
            data_type='frame',
            duration=config.duration,
            transform=None
        )

    # valid_dataset = BinnedSpikingSpeechCommands(
    #     config.datasets_path,
    #     config.n_bins,
    #     split='valid',
    #     data_type='frame',
    #     duration=config.duration
    # )
    test_dataset = BinnedSpikingSpeechCommands(
        config.datasets_path,
        config.n_bins,
        split='test',
        data_type='frame',
        duration=config.duration
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=pad_sequence_collate,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(dataset_seed + 0),
    )

    # valid_loader = DataLoader(
    #     valid_dataset,
    #     collate_fn=pad_sequence_collate,
    #     batch_size=config.batch_size,
    #     shuffle=False,
    #     num_workers=getattr(config, 'num_workers', 0),
    #     pin_memory=True,
    #     worker_init_fn=seed_worker,
    #     generator=make_torch_generator(dataset_seed + 1),
    # )

    test_loader = DataLoader(
        test_dataset,
        collate_fn=pad_sequence_collate,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(dataset_seed + 2),
    )

    # return train_loader, valid_loader, test_loader
    return train_loader, None, test_loader

# =========================================================
# SHD dataset wrappers
# =========================================================
class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
        self,
        root: str,
        n_bins: int,
        train: bool = None,
        data_type: str = 'event',
        frames_number: int = None,
        split_by: str = None,
        duration: int = None,
        custom_integrate_function: Callable = None,
        custom_integrated_frames_dir_name: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root, train, data_type, frames_number, split_by, duration,
            custom_integrate_function, custom_integrated_frames_dir_name,
            transform, target_transform
        )
        self.n_bins = n_bins

    def __getitem__(self, index: int):
        if self.data_type == 'event':
            events = {
                't': self.h5_file['spikes']['times'][index],
                'x': self.h5_file['spikes']['units'][index]
            }
            label = self.h5_file['labels'][index]

            if self.transform is not None:
                events, label = self.transform(events, label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[index], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[index]

            binned_len = frames.shape[1] // self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len), dtype=np.float32)
            for t in range(binned_len):
                binned_frames[:, t] = frames[:, self.n_bins * t: self.n_bins * (t + 1)].sum(axis=1)

            if self.transform is not None:
                binned_frames, label = self.transform(binned_frames, label)

            return binned_frames, label

        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")


class BinnedSpikingSpeechCommands(SpikingSpeechCommands):
    def __init__(
        self,
        root: str,
        n_bins: int,
        split: str = 'train',
        data_type: str = 'event',
        frames_number: int = None,
        split_by: str = None,
        duration: int = None,
        custom_integrate_function: Callable = None,
        custom_integrated_frames_dir_name: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root, split, data_type, frames_number, split_by, duration,
            custom_integrate_function, custom_integrated_frames_dir_name,
            transform, target_transform
        )
        self.n_bins = n_bins

    def __getitem__(self, index: int):
        if self.data_type == 'event':
            events = {
                't': self.h5_file['spikes']['times'][index],
                'x': self.h5_file['spikes']['units'][index]
            }
            label = self.h5_file['labels'][index]

            if self.transform is not None:
                events, label = self.transform(events, label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[index], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[index]

            binned_len = frames.shape[1] // self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len), dtype=np.float32)
            for t in range(binned_len):
                binned_frames[:, t] = frames[:, self.n_bins * t: self.n_bins * (t + 1)].sum(axis=1)

            if self.transform is not None:
                binned_frames, label = self.transform(binned_frames, label)

            return binned_frames, label

        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")


# =========================================================
# NTIDIGITS
# =========================================================
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url


class NTIDIGITS(VisionDataset):
    """
    N-TIDIGITS dataset.
    """

    resource = "https://dl.dropbox.com/s/vfwwrhlyzkax4a2/n-tidigits.hdf5?dl=1"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        duration: float = 1.0,
    ) -> None:
        super(NTIDIGITS, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        self.duration = duration

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        data_file = self.training_file if self.train else self.test_file

        with open(os.path.join(self.processed_folder, data_file), 'rb') as pickled_file:
            self.data, self.targets = torch.load(pickled_file)

    def __getitem__(self, index: int):
        utterance, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            utterance, target = self.transform(utterance, target)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return utterance, target

    def __len__(self):
        return len(self.data)

    @property
    def training_file(self):
        return f'training{int(self.duration)}.pt'

    @property
    def test_file(self):
        return f'test{int(self.duration)}.pt'

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self) -> bool:
        return (
            os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
            os.path.exists(os.path.join(self.processed_folder, self.test_file))
        )

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        download_url(self.resource, root=self.raw_folder, filename='n-tidigits.hdf5')

        print('Processing...')

        training_set = self.read_event_audio_files(os.path.join(self.raw_folder, 'n-tidigits.hdf5'), True)
        test_set = self.read_event_audio_files(os.path.join(self.raw_folder, 'n-tidigits.hdf5'), False)

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

    def read_event_audio_files(self, path: str, train: bool):
        ntidigits = h5py.File(path, 'r')
        dataset_group = 'train' if train else 'test'
        spikes_array, labels_list = [], []
        image_time = 1000.0
        number_of_timesteps = int(image_time / self.duration)

        bins = np.arange(0.0, image_time, self.duration, dtype=np.int16)

        for sample_key in ntidigits[dataset_group + '_addresses']:
            suffix = sample_key.rpartition('-')[2]
            if len(suffix) == 1 and (suffix.isdigit() or suffix == 'z'):
                neuron_indexes = torch.from_numpy(ntidigits[dataset_group + '_addresses'][sample_key][...])
                timestamps_ms = torch.from_numpy(ntidigits[dataset_group + '_timestamps'][sample_key][...]) * 1000
                label = suffix

                bin_indexes = torch.tensor(np.digitize(timestamps_ms, bins) - 1, dtype=torch.int32)

                sparse_indexes = torch.stack([bin_indexes, neuron_indexes], dim=0)
                sparse_values = torch.ones((neuron_indexes.size()), dtype=torch.uint8)
                spikes = torch.sparse_coo_tensor(
                    sparse_indexes, sparse_values, (number_of_timesteps, 64)
                ).to_dense()

                spikes_array.append(torch.where(spikes > 1, torch.ones_like(spikes), spikes))
                labels_list.append(0 if label == 'z' else int(label))

        return spikes_array, torch.ByteTensor(labels_list)


def ntidigits_collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0).float()   # (B, T, N)
    ys = torch.tensor(ys, dtype=torch.long)
    x_len = torch.full((xs.size(0),), xs.size(1), dtype=torch.long)
    return xs, ys, x_len


def NTIDIGITS_dataloaders(config, dataset_seed=None):
    dataset_seed = resolve_dataset_seed(config, dataset_seed)
    print("Debug:", config.datasets_path)

    if config.use_aug:
        augs = Augs(config)
        print("Data Augmentations used:", augs.list_augs())
        train_dataset = NTIDIGITS(
            root=config.datasets_path,
            train=True,
            download=getattr(config, 'allow_download', False),
            duration=config.duration,
            transform=augs,
        )
    else:
        train_dataset = NTIDIGITS(
            root=config.datasets_path,
            train=True,
            download=False,
            duration=config.duration,
        )

    test_dataset = NTIDIGITS(
        root=config.datasets_path,
        train=False,
        download=False,
        duration=config.duration,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
        collate_fn=ntidigits_collate_fn,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(dataset_seed + 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 0),
        pin_memory=True,
        collate_fn=ntidigits_collate_fn,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(dataset_seed + 1),
    )

    return train_loader, test_loader