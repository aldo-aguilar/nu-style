import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation


def getData(batch_size, imsize):
    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Resize(imsize),
                                      transforms.ToTensor()
                                      ])
    )

    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    images, _ = next(iter(dataloader))

    mid = int(images.shape[0] / 2)
    # Real Images Dataset 1
    real = images[:mid]

    # Rotated images Dataset 2
    rot = images[mid:]
    rot = scipy.ndimage.interpolation.rotate(rot, 90, axes=(2, 3))
    rot = torch.tensor(rot)
    return real, rot


def show(img, title=None):
    img = torch.squeeze(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    if title:
        plt.title(title)
    plt.pause(4)
