import torch
import torchvision
import numpy as np
from pathlib import Path

import default_param

default_param.MATPLOTLIB_NO_GUI = False
from semiseg.datasets.dataloader import CityscapesLoader
from semiseg.datasets.datautils import CityscapesUtils
from semiseg.networks.fcn import FCN8s, FCN32s, FCN16s
from semiseg.networks.resnet import Resnet50
from semiseg.models.seg_model import SegmentationModel
from semiseg.models.self_sup_model import SelfSupervisedModel
from training import training_utils, train

directory_name = Path(__file__).parents[1].resolve()
output_directory = directory_name / "semantic_segmentation" / "outputs"
output_directory.mkdir(parents=True, exist_ok=True)


def get_dataloader(mode="supervised"):
    cityscapes = CityscapesLoader(label_percent='100%').get_cityscapes_loader(mode=mode)
    return cityscapes


def get_model():
    cityscapes = get_dataloader()
    cityscapes_utils = CityscapesUtils()
    num_classes = cityscapes.utils.num_classes + 1
    num_features = [256, 512, 512]
    base = Resnet50(pretrained=False)
    fcn8s = FCN8s(base, num_classes)
    optim = torch.optim.Adam(fcn8s.parameters())
    model = SegmentationModel(fcn8s, cityscapes, optim)
    return model


def debug_FCN():
    cityscapes = get_dataloader()
    batch_imgs, batch_targets = next(iter(cityscapes['train']))
    training_utils.plot_images(batch_imgs, batch_targets, title='Predictions')


def debug_self_supervised_model():
    cityscapes = get_dataloader(mode='self-supervised')
    (batch_imgs, tf_imgs, seeds), _ = next(iter(cityscapes['train']))
    preds = None
    training_utils.plot_images(batch_imgs, tf_imgs, preds, title='Predictions')


if __name__ == '__main__':
    debug_FCN()
    debug_self_supervised_model()
