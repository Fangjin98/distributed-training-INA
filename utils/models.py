import re
import os
import torch
import torch.nn.functional as F
import torch.nn as nn

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def get_model(model_name):
    """
    :param model_name: alexnet vgg11 vgg16 vgg19 resnet50 resnet101 resnet 152
    :return: model instance
    """
    model = torch.hub.load(CURRENT_PATH + '/../data/pytorch/vision-0.10.0',
                           model_name, source='local', pretrained=True)
    model.eval()
    return model
