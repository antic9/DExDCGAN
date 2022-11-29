from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import model
import datetime
from parameters import HPS



# 乱数の固定
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



# dataset
dataroot = "./Dataset/cnn_opt_train"
# dataloader
workers = 2
# batch_size
batch_size = 1280
# Resize image
image_size = 64
# RGB
nc = 3
# Latent space dim
nz = 256
# Number of feature maps in generator
ngf = 128
# Number of feature maps in discriminator
ndf = 128
# Number of training epochs
num_epochs = 100
# Learning rate
lr = 0.002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

def directory():
    save_id = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    runs_dir = Path(f"runs")
    root_dir = runs_dir / f"{save_id}"
    img_dir = root_dir / "images"
    chk_dir = root_dir / "checkpoints"
    log_dir = root_dir / "logs"

    runs_dir.mkdir(exist_ok=True)
    root_dir.mkdir(exist_ok=True)
    chk_dir.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    return img_dir, chk_dir, log_dir

# custom weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def create_dataset():
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    return dataloader

def save_modelinfo(netG, netD, _pth):
    with open(_pth / f"info.txt","w") as f:
        f.write(f"Generator:\n{netG}\nDiscriminator:\n{netD}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--task', type=str, default='cnn')
    args = parser.parse_args()

    # Decide which device we want to run on
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    dataloader = create_dataset()
    img_dir, chk_dir, log_dir = directory()
    
    cfg = HPS[args.task]
    # # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    # Create the generator
    netG = model.Generator(cfg).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(weights_init)

    # Create the Discriminator
    netD = model.Discriminator(cfg).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netD.apply(weights_init)

    # Save Model Info
    save_modelinfo(netG,netD,log_dir)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create latent vector
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # real and fake labels
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    iters = 0

    # Training

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Loss
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save Losses
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Generate image
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                plt.figure(figsize=(8,8))
                plt.axis("off")
                plt.title("Fake Images (iteration " + str(iters) +")")
                plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
                if epoch % 10 == 0:
                    plt.savefig(img_dir / f"fake-{str(epoch).zfill(4)}.png")
                    torch.save(netG.state_dict(), chk_dir / f'{str(epoch).zfill(4)}.pth')
            iters += 1

