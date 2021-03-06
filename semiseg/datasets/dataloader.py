from typing import Dict
import os
import torch
from torchvision import datasets
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from default_param import CITYSCAPES_PATH, batch_size, num_workers
from semiseg.datasets.dataset import CityscapesDataset
from augmentations import get_tfms, get_self_supervised_tfms


class CityscapesLoader:
    """
    prepares the cityscapes Dataloader
    """

    def __init__(self, label_percent: str = '100%') -> None:
        """

        :param label_percent: Percentage of cityscapes labels to be used must be suffixed with %
        """
        label_percent = min(float(label_percent[:-1]), 100.0)
        self.prepare_dataset(label_percent)

    def _prepare_dataset(self, label_percent: int = 100) -> None:
        """
        Constructs all 3 datasets(train, test, val) using CityScapes Dataset class
        :param label_percent:  Percentage of cityscapes labels to be used
        :return: None
        """
        data_fine = {phase: datasets.Cityscapes(CITYSCAPES_PATH, split=phase, mode='fine',
                                                target_type='semantic') for phase in ['train', 'test', 'val']}
        if 'gtCoarse' in os.listdir(CITYSCAPES_PATH):
            data_coarse = {phase: datasets.Cityscapes(CITYSCAPES_PATH, split=phase, mode='coarse',
                                                      target_type='semantic') for phase in ['train_extra', 'val']}

        else:
            data_coarse = {
                'train_extra': [],
                'val': data_fine['val']
            }

        img_tfms, target_tfms = get_tfms()
        self_supervised_tfms = get_self_supervised_tfms()

        self.cityscapes = {phase: CityscapesDataset(data_fine[phase],
                                                    label_percent=label_percent if phase == 'train' else 100,
                                                    transform=img_tfms[phase],
                                                    target_transform=target_tfms[phase])
                           for phase in ['train', 'val']}
        self.cityscapes['self-supervised'] = {
            phase: CityscapesDataset([data_fine['train'], data_fine['test']]
                                     if phase == 'train' else data_coarse['val'],
                                     label_percent=label_percent if phase == 'train' else 100,
                                     transform=self_supervised_tfms,
                                     mode='self-supervised')
            for phase in ['train', 'val']

        }

    def get_cityscapes_loader(self, batch_size: int = batch_size, num_workers: int = num_workers,
                              mode: str = 'supervised') -> Dict[str, torch.utils.data.DataLoader]:
        """

        :param batch_size: no of batches
        :param num_workers: no of parallel workers for dataloading on CPU
        :param mode: 'self-supervised' or'supervised' dataloader
        :return: Dictionary of train and val data loader

        """

        data = self.cityscapes if mode == 'supervised' else self.cityscapes['self-supervised']

        cityscapes_loader = {x: torch.utils.data.DataLoader(data[x], batch_size=batch_size,
                                                            sampler=RandomSampler(data[x]) if x == 'train'
                                                            else SequentialSampler(data[x]),
                                                            drop_last=bool(mode == 'supervised' and x == 'train'),
                                                            num_workers=num_workers,
                                                            pin_memory=True)
                             for x in ['train', 'val']}
        return cityscapes_loader
