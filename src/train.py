"""
Majority of code taken from here:
https://github.com/yagudin/PyTorch-deep-photo-styletransfer
"""

# Training building blocks
from src.handlers import *
from src.dataset import *
from src.closed_form_matting import compute_laplacian
import torch
import torchvision.models as models
import os


def main():
    device = 'cpu'
    imsize = 128

    os.chdir(os.path.dirname(os.getcwd()) + r'\examples')

    index = 1
    style_img = image_loader('style/tar{}.png'.format(index), imsize).to(device, torch.float)
    content_img = image_loader('input/in{}.png'.format(index), imsize).to(device, torch.float).clone()

    os.chdir(os.path.dirname(os.getcwd()) + r'\images')
    if os.path.exists('img.pt'):
        tensor = torch.load('img.pt')
        imshow(content_img)
        imshow(style_img)
        imshow(tensor)
        return

    os.chdir(os.path.dirname(os.getcwd()) + r'\examples')

    # Model
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    style_masks, content_masks = masks_loader(
        'segmentation/tar{}.png'.format(index),
        'segmentation/in{}.png'.format(index),
        imsize)

    laplacian = compute_laplacian(content_img.detach().numpy().transpose(0, 2, 3, 1).squeeze())

    # Hyper Parameters
    epochs = 300  # 1000
    print_count = 5
    style_weight = 10000000
    reg_weight = 1000
    content_weight = 1
    content_lyrs = ['conv4']
    style_lyrs = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

    print('Building the model')
    model, style_losses, content_losses = get_style_model_and_loss(vgg, vgg_normalization_mean, vgg_normalization_std,
                                                                   style_img, content_img, style_lyrs, content_lyrs,
                                                                   style_masks, content_masks)

    print(model)

    optimizer = get_input_optimizer(content_img)

    print('Optimizing...')
    i = 0
    while i <= epochs:
        content_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(content_img)
        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight / 5

        loss = style_score + content_score
        # loss.backward()

        # New loss shenanigans
        reg_loss, reg_grad = regularization_grad(content_img, laplacian)
        # not sure why this would be needed
        # reg_grad_tensor = torch.tensor(reg_grad)
        # print('Shape:', reg_grad_tensor.shape)
        # content_img.grad += reg_weight * reg_grad_tensor
        loss += reg_loss * reg_weight
        loss.backward()

        i += 1
        if i % (epochs / print_count) == 0:
            print(f'Epoch {i}')
            print(f'Style Loss: {style_score.item()} Content Loss {content_score.item()} Regularization Loss {reg_loss.item()}')
            print()
        optimizer.step(lambda: style_score + content_score + reg_loss)

    content_img.data.clamp_(0, 1)
    os.chdir(os.path.dirname(os.getcwd()) + r'\images')
    torch.save(content_img, 'img.pt')
    imshow(content_img)


if __name__ == '__main__':
    main()
