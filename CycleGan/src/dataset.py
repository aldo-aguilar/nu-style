import torch
import numpy as np
import matplotlib.pyplot as plt


def show(img):
    img = torch.squeeze(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.pause(4)
