# Data handling
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms


def image_loader(image_name, imsize, device='cpu'):
    image = Image.open(image_name)
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    image = loader(image)
    print('Images shape initial', image.shape)
    image = image.unsqueeze(0)  # gets right dimension at particular size
    print('Image shape after unsqueeze', image.shape)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # clone the tensor, prevent pass by ref
    image = torch.squeeze(image)  # index 0 is the batch dimension
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if not title:
        plt.title(title)
    plt.pause(10)  # pausing to update the plots
