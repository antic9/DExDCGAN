import numpy as np

import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision.utils import save_image
import argparse
import model
from parameters import HPS


parser = argparse.ArgumentParser()
parser.add_argument('--load-path', type=str, default=None)
parser.add_argument('--task', type=str, default='cnn')

args = parser.parse_args()

cfg = HPS[args.task]
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

netG = model.Generator(cfg).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

netG.load_state_dict(torch.load(args.load_path))

noise = np.zeros((64,cfg.nz,1,1))
noise_index = 1
scale = np.arange(33,528,step=66)
x = [i-3.5 for i in range(8)]
y = [3.5-i for i in range(8)]
z_shape = 2
for i in range(noise_index):
  j = int(i/8)
  #noise [i][14] = 1.3389836870156424
  #noise [i][15] = -2.5115570258698128
  noise [i][16] = i % 8 - 3.5
  noise [i][17] = 3.5 - j % 8 

fixed_noise = torch.tensor(noise, dtype=torch.float, device=device)

print(fixed_noise.shape)

with torch.no_grad():
  fake = netG(fixed_noise).detach().cpu()
# im = np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0))
# save_image(im, "decode.png")

# print()
plt.figure(figsize=(10,10))
#plt.axis("off")
#plt.title("Fake Images")
plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))
plt.xticks(scale, x)
plt.yticks(scale, y)
plt.xlabel("z1")
plt.ylabel("z2")
plt.savefig("decode.png")