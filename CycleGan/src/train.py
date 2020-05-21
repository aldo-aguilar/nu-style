import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import scipy.ndimage.interpolation

import src.model as m
import src.dataset as d
import os


def main():
    os.chdir(os.path.dirname(os.getcwd()) + r'\images')
    if os.path.exists('img1.pt') and os.path.exists('img2.pt'):
        tensor1 = torch.load('img1.pt')
        tensor2 = torch.load('img2.pt')
        d.show(tensor1)
        d.show(tensor2)
        return

    os.chdir(os.path.dirname(os.getcwd()) + r'\src')
    # Training Params
    torch.manual_seed(1)
    lr = 0.0002
    batch_size = 10
    epochs = 1000  # 00  # 0
    device = 'cpu'
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Network params
    imsize = 28  # img sz is 28x28

    # GAN for approximating A Distribution

    genAB = m.Generator(color_channels=1)
    disB = m.Discriminator(color_channels=1)
    genBA = m.Generator(color_channels=1)
    disA = m.Discriminator(color_channels=1)

    train_set = torchvision.datasets.FashionMNIST(
        root='./data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.Resize(imsize),
                                      transforms.ToTensor()
                                      ])
    )

    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # Assembling the training data from 2 domains
    images, _ = next(iter(dataloader))

    mid = int(images.shape[0] / 2)
    # Real Images Dataset 1
    real = images[:mid]

    # Rotated images Dataset 2
    rot = images[mid:]
    rot = scipy.ndimage.interpolation.rotate(rot, 90, axes=(2, 3))
    rot = torch.tensor(rot)

    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # clone the tensor, prevent pass by ref
        image = torch.squeeze(image)  # index 0 is the batch dimension
        unloader = transforms.ToPILImage()
        image = unloader(image)
        plt.imshow(image)
        if not title:
            plt.title(title)
        plt.pause(4)  # pausing to update the plots

    genAB.apply(m.weights_init)
    genBA.apply(m.weights_init)
    disA.apply(m.weights_init)
    disB.apply(m.weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup optimizers
    optimizerGAB = optim.Adam(genAB.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDA = optim.Adam(disA.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerGBA = optim.Adam(genBA.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDB = optim.Adam(disB.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_listA = []
    img_listB = []

    GAB_losses = []
    DB_losses = []
    GBA_losses = []
    DA_losses = []

    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(epochs):

        # real = A
        # rot = B

        # Train disA on real A
        # Train disB on real B
        # Train disB on genAB - backwards disB
        # Train disA on genBA - backwards disA
        # Backwards genAB
        # Backwards genBA

        disA.zero_grad()
        # Format batch
        A = real.to(device)
        batch_size = A.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        # Forward pass real batch through D
        output = disA(A).view(-1)
        # Calculate loss on all-real batch
        errDA_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errDA_real.backward()
        DA_x = output.mean().item()

        disB.zero_grad()
        # Format batch
        B = rot.to(device)
        batch_size = B.size(0)
        label = torch.full((batch_size,), real_label, device=device)
        # Forward pass real batch through D
        output = disB(B).view(-1)
        # Calculate loss on all-real batch
        errDB_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errDB_real.backward()
        DB_x = output.mean().item()

        B_fake = genAB(A)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = disB(B_fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errDB_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errDB_fake.backward(retain_graph=True)
        DB_GAB_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errDB = errDB_real + errDB_fake
        # Update D
        optimizerDB.step()

        A_fake = genBA(B_fake)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = disA(A_fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errDA_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errDA_fake.backward(retain_graph=True)
        DA_GBA_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errDA = errDA_real + errDA_fake
        # Update D
        optimizerDA.step()

        genAB.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = disB(B_fake).view(-1)
        # Calculate G's loss based on this output
        errGAB = criterion(output, label)
        # Calculate gradients for G
        errGAB.backward(retain_graph=True)
        DB_GAB_z2 = output.mean().item()
        # Update G
        optimizerGAB.step()

        genBA.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = disA(A_fake).view(-1)
        # Calculate G's loss based on this output
        errGBA = criterion(output, label)
        # Calculate gradients for G
        errGBA.backward()
        DA_GBA_z2 = output.mean().item()
        # Update G
        optimizerGBA.step()

        # Output training stats
        if epoch % (epochs/10) == 0:
            print('[%d/%d]\tLoss_DA: %.4f\tLoss_GBA: %.4f\tDA(x): %.4f\tDA(GBA(z)): %.4f / %.4f\tLoss_DB: '
                  '%.4f\tLoss_GAB: %.4f\tDB(x): %.4f\tDB(GAB(z)): %.4f / %.4f '
                  % (epoch, epochs,  # i, len(real),
                     errDA.item(), errGBA.item(), DA_x, DA_GBA_z1, DA_GBA_z2,
                     errDB.item(), errGAB.item(), DB_x, DB_GAB_z1, DB_GAB_z2))

        # Save Losses for plotting later
        GAB_losses.append(errGAB.item())
        DB_losses.append(errDB.item())
        GBA_losses.append(errGBA.item())
        DA_losses.append(errDA.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % (epochs/4) == 0) or (epoch == epochs - 1):
            with torch.no_grad():
                fakeB = genAB(A).detach().to(device)
                fakeA = genBA(B).detach().to(device)
            img_listB.append(vutils.make_grid(fakeB, padding=2, normalize=True))
            img_listA.append(vutils.make_grid(fakeA, padding=2, normalize=True))

        iters += 1

    img_list = zip(img_listA, img_listB)
    length = len(img_listA)

    for i, data in enumerate(img_list):
        img1, img2 = data
        d.show(img1)
        d.show(img2)
        if i == length - 1:
            os.chdir(os.path.dirname(os.getcwd()) + r'\images')
            torch.save(img1, 'img1.pt')
            torch.save(img2, 'img2.pt')

    # for img in A:
    #     imshow(img)
    #     B_fake = genAB(torch.unqimg).detach().to(device)
    #     show(B_fake)
    #     A_fake = genBA(B_fake).detach().to(device)
    #     show(A_fake)


if __name__ == '__main__':
    main()
