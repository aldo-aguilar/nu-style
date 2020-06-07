# Model building
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms


def _gram_matrix(x):
    a, b, c, d = x.size()
    # a batch size
    # b number of feature maps
    # c, d | dimensions of a feature map (N=c*d)

    features = x.view(a * b, c * d)  # resizing feature map of the layer
    G = torch.mm(features, features.t())  # gram product calculation
    # normalization here
    G_normalized = G.div(a * b * c * d)
    # we can play with results non-normalized to see the impact on the output
    return G_normalized


# Style Loss: like content loss, uses MSE between two gram matricies
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # .detach means that this tensor does not need grad
        self.target = _gram_matrix(target_feature).detach()

    def forward(self, x):
        G = _gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1).detach()
        self.std = std.view(-1, 1, 1).detach()

    def forward(self, img):
        # normalize the image
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLossAug(nn.Module):
    def __init__(self, target, style, content):
        super(StyleLossAug, self).__init__()
        shape = target.shape
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((shape[2], shape[3])), transforms.ToTensor()])
        loader = lambda mask: transform(torch.squeeze(mask.int())).unsqueeze(0)
        self.masks = []
        for mask in content:
            self.masks.append(loader(mask.detach()))
        self.targets = []
        for mask in style:
            gram = _gram_matrix(target * loader(mask)).detach()
            self.targets.append(gram)

    def forward(self, x):
        loss = 0
        for mask, target in zip(self.masks, self.targets):
            gram = _gram_matrix(x * mask.detach())
            loss += F.mse_loss(gram, target.detach())
        self.loss = loss
        return x
