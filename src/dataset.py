# Data handling
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms

"""
Function which loads a specific image, and scales it accordingly 

Arguments:
    image_name - name of the image to be loaded
    imsize - dimensions to scale the image to
    device - device to send image to, default keyword argument is cpu
Returns:

"""

def image_loader(image_name, imsize, device='cpu'):
    image = Image.open(image_name)
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    image = loader(image)
    print('Images shape initial', image.shape)
    image = image.unsqueeze(0)  # gets right dimension at particular size
    print('Image shape after unsqueeze', image.shape)
    return image.to(device, torch.float)


"""
Function which takes an image as a tensor and plots the image using matplotlib

Arguments:
    tensor - image tensor to be plotted
    title - title of the plot to be generated
"""
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # clone the tensor, prevent pass by ref
    image = torch.squeeze(image)  # index 0 is the batch dimension
    unloader = transforms.ToPILImage()
    image = unloader(image)
    plt.imshow(image)
    if not title:
        plt.title(title)
    plt.pause(10)  # pausing to update the plots
