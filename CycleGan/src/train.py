import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils

import src.model as m
import src.dataset as d
import os


def main():
    # Training Params
    torch.manual_seed(1)
    lr = 0.0002
    batch_size = 10
    epochs = 100
    device = 'cpu'
    beta1 = 0.5
    lambda1 = 10
    imsize = 28  # in case we want to change it

    # Models
    genAB = m.Generator(color_channels=1)
    disB = m.Discriminator(color_channels=1)
    genBA = m.Generator(color_channels=1)
    disA = m.Discriminator(color_channels=1)

    # Data
    real, rot = d.getSingleData(batch_size, imsize)

    # Possible Quick Load
    os.chdir(os.path.dirname(os.getcwd()) + r'\images')
    if os.path.exists('img1.pt'):
        img1 = torch.load('img1.pt')
        img2 = torch.load('img2.pt')
        d.show(img1, 'last generated real')
        d.show(img2, 'last generated fake')
        os.chdir(os.path.dirname(os.getcwd()) + r'\models')
        if os.path.exists('genAB.pt'):
            genAB.load_state_dict(torch.load('genAB.pt'))
            genAB.eval()
            genBA.load_state_dict(torch.load('genBA.pt'))
            genBA.eval()
            A = real.to(device)
            fakeB = genAB(A).detach().to(device)
            fakeA = genBA(fakeB).detach().to(device)
            d.show(vutils.make_grid(A, padding=2, normalize=True), title='real')
            d.show(vutils.make_grid(fakeB, padding=2, normalize=True), title='generated fake')
            d.show(vutils.make_grid(fakeA, padding=2, normalize=True), title='generated real')
            return

    os.chdir(os.path.dirname(os.getcwd()) + r'\src')

    # Apply normalization
    genAB.apply(m.weights_init)
    genBA.apply(m.weights_init)
    disA.apply(m.weights_init)
    disB.apply(m.weights_init)

    # Initialize Loss functions
    criterion = nn.BCELoss()  # In paper, it's MSELoss; but we found its not the great, prolly good for large data
    cyclic = nn.L1Loss()

    # Label representation (not tensors)
    real_label = 1
    fake_label = 0

    # Setup optimizers
    optimizerGAB = optim.Adam(genAB.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDA = optim.Adam(disA.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerGBA = optim.Adam(genBA.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDB = optim.Adam(disB.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_listA = []
    img_listB = []

    iters = 0
    batch_size = int(batch_size/2)

    print("Starting Training Loop...")
    for epoch in range(epochs):

        # Parameters
        A = real.to(device)
        B = rot.to(device)
        rlabel = torch.full((batch_size,), real_label, device=device)
        flabel = torch.full((batch_size,), fake_label, device=device)

        # Train Generators
        A_fake = genBA(B)
        B_fake = genAB(A)
        A_rev = genBA(B_fake)
        B_rev = genAB(A_fake)
        A_rev_err = cyclic(A_rev, A)
        B_rev_err = cyclic(B_rev, B)

        output = disB(B_fake).view(-1)
        errGAB = criterion(output, rlabel)
        DB_GAB_z2 = output.mean().item()

        output = disA(A_fake).view(-1)
        errGBA = criterion(output, rlabel)
        DA_GBA_z2 = output.mean().item()

        G_loss = errGAB + errGBA + B_rev_err * lambda1 + A_rev_err * lambda1

        # Train Discriminators
        output = disA(A).view(-1)
        errDA_real = criterion(output, rlabel)
        DA_x = output.mean().item()

        output = disB(B).view(-1)
        errDB_real = criterion(output, rlabel)
        DB_x = output.mean().item()

        output = disB(B_fake.detach()).view(-1)
        errDB_fake = criterion(output, flabel)
        DB_GAB_z1 = output.mean().item()
        errDB = (errDB_real + errDB_fake) * 0.5

        output = disA(A_fake.detach()).view(-1)
        errDA_fake = criterion(output, flabel)
        DA_GBA_z1 = output.mean().item()
        errDA = (errDA_real + errDA_fake) * 0.5

        # Update Generators
        genAB.zero_grad()
        genBA.zero_grad()
        G_loss.backward()
        optimizerGAB.step()
        optimizerGBA.step()

        # Update Discriminators
        disA.zero_grad()
        disB.zero_grad()
        errDA.backward()
        errDB.backward()
        optimizerDA.step()
        optimizerDB.step()

        if epoch % (epochs / 10) == 0:
            print('[%d/%d]\tLoss_DA: %.4f\tLoss_GBA: %.4f\tDA(x): %.4f\tDA(GBA(z)): %.4f / %.4f\tLoss_DB: '
                  '%.4f\tLoss_GAB: %.4f\tDB(x): %.4f\tDB(GAB(z)): %.4f / %.4f '
                  % (epoch, epochs,
                     errDA.item(), errGBA.item(), DA_x, DA_GBA_z1, DA_GBA_z2,
                     errDB.item(), errGAB.item(), DB_x, DB_GAB_z1, DB_GAB_z2))

        if (iters % (epochs / 4) == 0) or (epoch == epochs - 1):
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
        d.show(vutils.make_grid(A, padding=2, normalize=True), title='real real')
        d.show(img1, title='fake real')
        d.show(vutils.make_grid(B, padding=2, normalize=True), title='real fake')
        d.show(img2, title='fake fake')
        if i == length - 1:
            os.chdir(os.path.dirname(os.getcwd()) + r'\images')
            torch.save(img1, 'img1.pt')
            torch.save(img2, 'img2.pt')

    os.chdir(os.path.dirname(os.getcwd()) + r'\models')
    torch.save(genAB.state_dict(), 'genAB.pt')
    torch.save(genBA.state_dict(), 'genBA.pt')
    torch.save(disA.state_dict(), 'disA.pt')
    torch.save(disB.state_dict(), 'disB.pt')


def linear_test():
    # Training Params
    torch.manual_seed(1)
    lr = 0.0002
    batch_size = 10
    epochs = 100
    device = 'cpu'
    beta1 = 0.5
    lambda1 = 10
    imsize = 28  # in case we want to change it

    # Models
    genAB = m.GeneratorLinear()
    disB = m.Discriminator(color_channels=1)
    genBA = m.GeneratorLinear()
    disA = m.Discriminator(color_channels=1)

    # Data
    real, rot = d.getSingleData(batch_size, imsize)

    # Possible Quick Load
    os.chdir(os.path.dirname(os.getcwd()) + r'\images')
    if os.path.exists('imgL1.pt'):
        img1 = torch.load('imgL1.pt')
        img2 = torch.load('imgL2.pt')
        d.show(img1, 'last generated real')
        d.show(img2, 'last generated fake')
        os.chdir(os.path.dirname(os.getcwd()) + r'\models')
        if os.path.exists('genLAB.pt'):
            genAB.load_state_dict(torch.load('genLAB.pt'))
            genAB.eval()
            genBA.load_state_dict(torch.load('genLBA.pt'))
            genBA.eval()
            A = real.to(device)
            fakeB = genAB(A).detach().to(device)
            fakeA = genBA(fakeB).detach().to(device)
            d.show(vutils.make_grid(A, padding=2, normalize=True), title='real')
            d.show(vutils.make_grid(fakeB, padding=2, normalize=True), title='generated fake')
            d.show(vutils.make_grid(fakeA, padding=2, normalize=True), title='generated real')
            return

    os.chdir(os.path.dirname(os.getcwd()) + r'\src')

    # Initialize Loss functions
    criterion = nn.BCELoss()  # In paper, it's MSELoss; but we found its not the great, prolly good for large data
    cyclic = nn.L1Loss()

    # Label representation (not tensors)
    real_label = 1
    fake_label = 0

    # Setup optimizers
    optimizerGAB = optim.Adam(genAB.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDA = optim.Adam(disA.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerGBA = optim.Adam(genBA.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerDB = optim.Adam(disB.parameters(), lr=lr, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_listA = []
    img_listB = []

    iters = 0
    batch_size = int(batch_size / 2)

    print("Starting Training Loop...")
    for epoch in range(epochs):

        # Parameters
        A = real.to(device)
        B = rot.to(device)
        rlabel = torch.full((batch_size,), real_label, device=device)
        flabel = torch.full((batch_size,), fake_label, device=device)

        # Train Generators
        A_fake = genBA(B)
        B_fake = genAB(A)
        A_rev = genBA(B_fake)
        B_rev = genAB(A_fake)
        A_rev_err = cyclic(A_rev, A)
        B_rev_err = cyclic(B_rev, B)

        output = disB(B_fake).view(-1)
        errGAB = criterion(output, rlabel)
        DB_GAB_z2 = output.mean().item()

        output = disA(A_fake).view(-1)
        errGBA = criterion(output, rlabel)
        DA_GBA_z2 = output.mean().item()

        G_loss = errGAB + errGBA + B_rev_err * lambda1 + A_rev_err * lambda1

        # Train Discriminators
        output = disA(A).view(-1)
        errDA_real = criterion(output, rlabel)
        DA_x = output.mean().item()

        output = disB(B).view(-1)
        errDB_real = criterion(output, rlabel)
        DB_x = output.mean().item()

        output = disB(B_fake.detach()).view(-1)
        errDB_fake = criterion(output, flabel)
        DB_GAB_z1 = output.mean().item()
        errDB = (errDB_real + errDB_fake) * 0.5

        output = disA(A_fake.detach()).view(-1)
        errDA_fake = criterion(output, flabel)
        DA_GBA_z1 = output.mean().item()
        errDA = (errDA_real + errDA_fake) * 0.5

        # Update Generators
        genAB.zero_grad()
        genBA.zero_grad()
        G_loss.backward()
        optimizerGAB.step()
        optimizerGBA.step()

        # Update Discriminators
        disA.zero_grad()
        disB.zero_grad()
        errDA.backward()
        errDB.backward()
        optimizerDA.step()
        optimizerDB.step()

        if epoch % (epochs / 10) == 0:
            print('[%d/%d]\tLoss_DA: %.4f\tLoss_GBA: %.4f\tDA(x): %.4f\tDA(GBA(z)): %.4f / %.4f\tLoss_DB: '
                  '%.4f\tLoss_GAB: %.4f\tDB(x): %.4f\tDB(GAB(z)): %.4f / %.4f '
                  % (epoch, epochs,
                     errDA.item(), errGBA.item(), DA_x, DA_GBA_z1, DA_GBA_z2,
                     errDB.item(), errGAB.item(), DB_x, DB_GAB_z1, DB_GAB_z2))

        if (iters % (epochs / 4) == 0) or (epoch == epochs - 1):
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
        d.show(vutils.make_grid(A, padding=2, normalize=True), title='real real')
        d.show(img1, title='fake real')
        d.show(vutils.make_grid(B, padding=2, normalize=True), title='real fake')
        d.show(img2, title='fake fake')
        if i == length - 1:
            os.chdir(os.path.dirname(os.getcwd()) + r'\images')
            torch.save(img1, 'imgL1.pt')
            torch.save(img2, 'imgL2.pt')

    os.chdir(os.path.dirname(os.getcwd()) + r'\models')
    torch.save(genAB.state_dict(), 'genLAB.pt')
    torch.save(genBA.state_dict(), 'genLBA.pt')
    torch.save(disA.state_dict(), 'disLA.pt')
    torch.save(disB.state_dict(), 'disLB.pt')


if __name__ == '__main__':
    main()
    # linear_test()
