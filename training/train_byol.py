import torch
import argparse
from typing import Dict
import importlib
import default_param
from semiseg.datasets.dataloader import CityscapesLoader
import json


def run_experiment(experiment_config: Dict, load_weights: bool, save_weights: bool) -> None:
    """

    param experiment_config:  Dictionary of the following form
        run a training experiment
     {
     "dataset": "cityscapes",
     "model": "SelfSupervisedModel",   # Type of the training model(SegmentationModel, SelfSupervisedModel)
     "network": "BYOL",                # "BYOL"
     "mode": "self-supervised",        # Type of training(self-supervised | supervised)
     "network_args": {"backbone":"Resnet50",   # "Resnet50" | "Resnet18" | "Resnet101" | "DRN50" etc
                      "pretrained": false,      # Whether the backbone model is pretrained
                      "target_momentum": 0.996  #Target momentum for byol target network
                      }
     "train_args":{"batch_size": 8,
                    "epochs": 400,
                     "log_to_tensorboard": false},
    "experiment_group":{}
}'
    :param experiment_config:
    :param load_weights: if true, load weights for the model from last run
    :param save_weights:if true, save model weights to semiseg/weights directory
    :return: None
    """
    default_training_argument = {'epochs': default_param.num_epochs, 'batch_size': default_param.batch_size,
                                 'num_workers': default_param.num_workers}

    training_args = {
        **default_training_argument,
        **experiment_config.get("training_args", {})
    }
    experiment_config['training_args'] = training_args
    experiment_config['experiment_group'] = experiment_config.get('experiment_group', None)
    print(f'Running experiment with configuration {experiment_config}')
    labels_percent = training_args.get('labels_percent', '100%')
    mode = experiment_config.get('mode', 'supervised')
    dataset_name = experiment_config['dataset'].lower()
    assert dataset_name in ['cityscapes'], 'The dataloader is only implemented for cityscapes dataset'
    data_loader = CityscapesLoader(label_percent=labels_percent).get_cityscapes_loader(
        batch_size=training_args['batch_size'],
        num_workers=training_args['num_workers'],
        mode=mode)
    models_module = importlib.import_module("sem_seg.models")
    model_class_ = getattr(models_module, experiment_config["model"])

    networks_module = importlib.import_module("sem_seg.networks")
    network_args = experiment_config.get("network_args", {})

    pretrained = network_args["pretrained"]
    backbone_class_ = getattr(networks_module, network_args["backbone"])
    base = backbone_class_(pretrained=pretrained)

    network_class_ = getattr(networks_module, experiment_config["network"])
    target_momentum = training_args.get("target_momentum", 0.996)
    network = network_class_(base, target_momentum=target_momentum)

    training_params = [*network.online_network.parameters()] + [*network.online_projector.parameters()] + [
        *network.predictor.parameters()]  # explicitly excluding the target network parameters
    optim = torch.optim.SGD(training_params, lr=3e-2, momentum=0.9, weight_decay=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.2,
                                                              patience=20, min_lr=1e-10, verbose=True)
    additional_identifier = get_additional_identifier(network_args["backbone"], pretrained, dataset_name,
                                                      labels_percent)
    model = model_class_(network, data_loader, optim, lr_scheduler=lr_scheduler,
                         additional_identifier=additional_identifier)

    if not training_args['log_to_tensorboard']:
        model.logToTensorboard = False
        model.add_text_to_tensorboard(json.dumps(experiment_config))

    if load_weights:
        model.load_weights()

    model.train(num_epochs=training_args["epochs"])

    if save_weights:
        model.save_weights()


def get_additional_identifier(backbone: str, pretrained: bool = False, dataset_name: str = '',
                              labels_percent: str = '100%') -> str:
    """
    Returns the additional_identifier added to the model name for efficient tracking of different experiments
    :param backbone: name of the backbone
    :param pretrained: Whether the backbone is pretrained
    :param dataset_name: Name of the training dataset
    :param labels_percent: % of labels used for training
    :return: additional identifier string
    """
    additional_identifier = backbone
    additional_identifier += '_pt' if pretrained else ''
    additional_identifier += '_' + labels_percent[:-1] if labels_percent and int(labels_percent[:-1]) < 100 else ''
    additional_identifier += '_ct' if dataset_name == 'cityscapes' else '_' + dataset_name
    return additional_identifier


def _parse_args():
    """ parse command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default=False, action="store_true",
                        help="If true, final weights will be stored in canonical, version-controlled location")
    parser.add_argument("--load", default=False, action="store_true",
                        help="If true, final weights will be loaded from canonical, version-controlled location")
    parser.add_argument("experiment_config", type=str,
                        help='Experiment JSON (\'{"dataset": "cityscapes", "model": "SegmentationModel",'
                             ' "network": "fcn8s"}\''
                        )
    args = parser.parse_args()
    return args


def main():
    """Run Experiment"""
    args = _parse_args()
    experiment_config = json.loads(args.experiment_config)
    run_experiment(experiment_config, args.load, args.save)


if __name__ == '__main__':
    main()
