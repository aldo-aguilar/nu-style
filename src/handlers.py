# Any handlers during training
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from .model import Normalization, ContentLoss, StyleLoss


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def get_style_model_and_loss(cnn, normalization_mean, normalization_std,
                             style_img, content_img, content_layers,
                             style_layers, device='cpu'):
    cnn = copy.deepcopy(cnn)

    # using the normalization class to normalize mean and std
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # list made for accumulating losses
    content_losses = []
    style_losses = []

    # using nn.Sequential because we assume that the cnn is already a
    # nn.Sequential, this lets us put modules to be activated sequentially
    model = nn.Sequential(normalization)

    number_of_conv = 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            number_of_conv += 1
            name = f'conv{number_of_conv}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{number_of_conv}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{number_of_conv}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{number_of_conv}'
        else:
            raise RuntimeError('Unrecongnized layer:'.format(layer.__class__.__name__))
        model.add_module(name, layer)

        if name in content_layers:
            # adding to the conetent loss
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f'content_loss_{number_of_conv}', content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # adding to the style loss
            target_features = model(style_img).detach()
            style_loss = StyleLoss(target_features)
            model.add_module(f'style_loss{number_of_conv}', style_loss)
            style_losses.append(style_loss)
    # trim off layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses
