import re
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as m

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def get_model(model_name, download=False):
    """
    :param model_name: alexnet vgg11 vgg16 vgg19 resnet50 resnet101 resnet152
    :return: model instance
    """
    model = None

    if model_name == 'resnet50':
        model = m.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        model = m.vgg16(pretrained=True)
    elif model_name == 'alexnet':
        model = m.alexnet(pretrained=True)

    if model is not None:
        model.eval()
    return model
