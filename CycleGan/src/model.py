import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, color_channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential()

        self.main.add_module("conv1", nn.Conv2d(color_channels, 64, 7, stride=1, padding=3))
        self.main.add_module("batch_norm1", nn.BatchNorm2d(64))
        self.main.add_module("relu1", nn.ReLU(True))

        self.main.add_module("conv2", nn.Conv2d(64, 128, 3, stride=2, padding=1))
        self.main.add_module("batch_norm2", nn.BatchNorm2d(128))
        self.main.add_module("relu2", nn.ReLU(True))

        self.main.add_module("conv30", nn.Conv2d(128, 256, 3, stride=2, padding=1))
        self.main.add_module("batch_norm30", nn.BatchNorm2d(256))
        self.main.add_module("relu30", nn.ReLU(True))

        # Based off https://towardsdatascience.com/overview-of-cyclegan-architecture-and-training-afee31612a2f
        # for i in range(1):
        #     self.main.add_module("conv3{}".format(i+1), nn.Conv2d(256, 256, 3, stride=2, padding=1))
        #     self.main.add_module("batch_norm3{}".format(i+1), nn.BatchNorm2d(256))
        #     self.main.add_module("relu3{}".format(i+1), nn.ReLU(True))

        self.main.add_module("conv4", nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1))
        self.main.add_module("batch_norm4", nn.BatchNorm2d(128))
        self.main.add_module("relu4", nn.ReLU(True))

        self.main.add_module("conv5", nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))
        self.main.add_module("batch_norm5", nn.BatchNorm2d(64))
        self.main.add_module("relu5", nn.ReLU(True))

        self.main.add_module("conv6", nn.Conv2d(64, color_channels, 7, stride=1, padding=3))

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, color_channels=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential()

        self.main.add_module("conv1", nn.Conv2d(color_channels, 64, 4, stride=2, padding=1))
        self.main.add_module("batch_norm1", nn.BatchNorm2d(64))
        self.main.add_module("relu1", nn.ReLU(True))

        self.main.add_module("conv2", nn.Conv2d(64, 128, 4, stride=2, padding=1))
        self.main.add_module("batch_norm2", nn.BatchNorm2d(128))
        self.main.add_module("relu2", nn.ReLU(True))

        self.main.add_module("conv3", nn.Conv2d(128, 256, 4, stride=2, padding=1))
        self.main.add_module("batch_norm3", nn.BatchNorm2d(256))
        self.main.add_module("relu3", nn.ReLU(True))

        self.main.add_module("conv4", nn.Conv2d(256, 512, 4, stride=1, padding=1))
        self.main.add_module("batch_norm4", nn.BatchNorm2d(512))
        self.main.add_module("relu4", nn.ReLU(True))

        self.main.add_module("conv5", nn.Conv2d(512, 1, 4, stride=1, padding=1))
        self.main.add_module("sigmoid", nn.Sigmoid())

    def forward(self, x):
        return self.main(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        print('I called conv')
    elif classname.find('batch_norm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        print('I called batch')
