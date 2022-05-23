import torch
import numpy as np
import default_param
from typing import Callable, List
from pathlib import Path

from base import Model
import eval_metrices

dirn_ame = Path(__file__).parents[1].resolve()
out_dir = dirn_ame / "outputs" / "class_IOUs"
out_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SegmentationModel(Model):
    """
    chld class for segmentation model from base model
    """

    def __init__(self, network: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim,
                 criterion: Callable = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 additional_identifer: str = ''):
        super().__init__(network, dataloader, optimizer, criterion, lr_scheduler, additional_identifer)

        if criterion is None:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=default_param.CITYSCAPES_IGNORED_INDEX)
        self.metrices = ['mIOU', 'pixel_accuracy']

    def store_per_class_iou(self):
        print('calculating per class mIOU')
        self.network.eval()
        self.network.to(device)
        c_matrix = 0
        for inputs, labels in self.dataloader['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outs = self.network(inputs)
            c_matrix += eval_metrices.get_confusion_matrix(outs, labels)
        class_iou = eval_metrices.per_class_iu(c_matrix)
        class_iou = np.round(class_iou, decimals=6)
        file_path = str(out_dir) + '/' + self.name + '.csv'
        np.savetxt(file_path, class_iou, delimeter=',', fmt='%.6f')
        print(f'per class IOU saved at {file_path}')
