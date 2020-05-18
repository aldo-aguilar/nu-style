# Training building blocks
from src.handlers import get_style_model_and_loss, get_input_optimizer
from src.dataset import image_loader, imshow
import torch
import torchvision.models as models
import os

"""
Less of training loop and more just running the model. This function takes in two images one being the image
to style, and the other being the image to take the style from. How it works is we load the vgg model from pytorch
(may build the modified vgg19) which is modified from a normal vgg 19 architecture for style transfer. We then set
parameters for the vgg normalized mean and standard deviation. We set convolutions for style and content, we need 
more convolutions on style layers, and we use various depths of convos in our architecture to learn with the style 
from more general features, to more specific. 

Next we optimize the model for the set number of epochs. While optimizing we track the content loss and style loss,
and multiply them by the set weights for each to get our loss for one epoch, we use a backward pass on the loss to 
calculate the gradient, and then update the optimizers parameters based on the gradient of our loss. Making sure to 
zero the gradient after each epoch. 

Parameters: 
    content_img - name of the image to act as the content image. This image must be located in the images
    folder.
    style_img - name of the image to act as the style image. This image must be located in the images
    folder.
    epochs - number of epochs to run optimization for
    style_weight - value of style_weight, can be thought of as how much we want to apply the style from 
    the style image to the content image
    content_weight - value of content_weight, can be thought of as how much we want to preserve the content
    image after we apply the style
    img_size - dimensions for the output image, currently only squared images, working on making an option
    for maintaing the content images size
    load_prior - bool which will load the prior results from this method if they are saved (img.pt)
    device - which device to use
Side Effects: 
    Generates the output image as a graph and saves the output in the file img.pt
"""
def training_loop(content_img, style_img, epochs, style_weight=1000000, content_weight=1, \
    img_size=128,load_prior=False, device='cpu'):

    # checking if we want to load the prior result, and if the prior result exist 
    if load_prior and os.path.exists('img.pt'):
        tensor = torch.load('img.pt')
        imshow(tensor)
        return

    # setting the device 
    device = device
    
    # Data
    path = os.path.join(os.getcwd(), 'images')
    style_img = image_loader(os.path.join(path, style_img), img_size)
    content_img = image_loader(os.path.join(path, content_img), img_size)


    # Model
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Hyper Parameters
    epochs = epochs
    style_weight = style_weight
    content_weight = content_weight
    
    print_count = 5
    content_lyrs = ['conv4']
    style_lyrs = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

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
            print(f'Style Loss: {style_score.item()} Content Loss: {content_score.item()} Total Loss: {loss}')
            print()
        optimizer.step(lambda: style_score + content_score)

    # displaying image
    content_img.data.clamp_(0, 1)
    torch.save(content_img, 'img.pt')
    imshow(content_img)
