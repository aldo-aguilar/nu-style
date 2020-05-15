# Training building blocks
from src.handlers import get_style_model_and_loss, get_input_optimizer
from src.dataset import image_loader, imshow
import torch
import torchvision.models as models
import os


def main():
    device = 'cpu'
    imsize = 128
    os.chdir(os.path.dirname(os.getcwd()) + r'\images')
    if os.path.exists('img.pt'):
        tensor = torch.load('img.pt')
        imshow(tensor)
        return
    # Data
    style_img = image_loader('picasso.jpg', imsize)
    content_img = image_loader('dancing.jpg', imsize)

    # Model
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Hyper Parameters
    epochs = 50
    print_count = 5
    style_weight = 1000000
    content_weight = 1
    content_lyrs = ['conv4']
    style_lyrs = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

    print('Building the model')
    model, style_losses, content_losses = get_style_model_and_loss(vgg, vgg_normalization_mean, vgg_normalization_std,
                                                                   style_img, content_img, content_lyrs, style_lyrs)
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
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()

        i += 1
        if i % (epochs/print_count) == 0:
            print(f'Epoch {i}')
            print(f'Style Loss: {style_score.item()} Content Loss {content_score.item()}')
            print()
        optimizer.step(lambda: style_score + content_score)

    content_img.data.clamp_(0, 1)
    torch.save(content_img, 'img.pt')
    imshow(content_img)


if __name__ == '__main__':
    main()
