import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, color_channels=3, resid_layers=6):
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

        for i in range(resid_layers):
            self.main.add_module("resid_block{}".format(i+1), ResidualBlock(256, i))

        self.main.add_module("conv4", nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1))
        self.main.add_module("batch_norm4", nn.BatchNorm2d(128))
        self.main.add_module("relu4", nn.ReLU(True))

        self.main.add_module("conv5", nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1))
        self.main.add_module("batch_norm5", nn.BatchNorm2d(64))
        self.main.add_module("relu5", nn.ReLU(True))

        self.main.add_module("conv6", nn.Conv2d(64, color_channels, 7, stride=1, padding=3))
        self.main.add_module("tanh", nn.Tanh())

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


class ResidualBlock(nn.Module):

    def __init__(self, features, index):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential()
        self.main.add_module("resid{}1".format(index + 1), nn.Conv2d(features, features, 3, stride=1, padding=1))
        self.main.add_module("batch_norm_resid{}1".format(index + 1), nn.BatchNorm2d(features))
        self.main.add_module("relu_resid{}".format(index + 1), nn.ReLU(True))
        self.main.add_module("resid{}2".format(index + 1), nn.Conv2d(features, features, 3, stride=1, padding=1))
        self.main.add_module("batch_norm_resid{}2".format(index + 1), nn.BatchNorm2d(features))

    def forward(self, x):
        return x + self.main(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('batch_norm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
