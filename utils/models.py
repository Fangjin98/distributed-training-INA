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

    if model is not None:
        model.eval()
    return model

    # if download is False:
    #     model_path=CURRENT_PATH+'/../data/models/'
    #     try:
    #         model= torch.load(model_path+model_name+'.pth')
    #     except IOError as e:
    #         print("ERROR: No corresponding model, please download first.")
    #     else:
    #         model.eval()
    #         return model

    # Download model, have some problem in edge401

    # fold_path=CURRENT_PATH+'/../data/pytorch/vision-0.10.0'
    # try:
    #     model = torch.hub.load(fold_path,model_name, source='local', pretrained=True)
    # except Exception as e:
    #     model=torch.hub.load('pytorch/vision:v0.10.0',model_name,pretrained=True)
