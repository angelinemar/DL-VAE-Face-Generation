# train_gan.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from read_celeba import *

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Models
z_dim = 128
netGen = models.Generator(z_dim).to(device)
netDis = models.Discriminator().to(device)
optGen = optim.Adam(netGen.parameters(), lr=4e-4, betas=(0.5, 0.999))
optDis = optim.Adam(netDis.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Output folders
model_name = "gan"
out_folder = f"out/{model_name}/"
os.makedirs(out_folder, exist_ok=True)
save_folder = f"save/{model_name}/"
os.makedirs(save_folder, exist_ok=True)

# Training loop
num_epochs = 20
z_fixed = torch.randn(36, z_dim, device=device)
print("Start Training ...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        x_real = data[0].to(device)
        batch_size = x_real.size(0)

        # Train Discriminator
        netDis.zero_grad()

        '''
        TODO: Compute probability of real data and construct labels
        d_real = 
        d_label = 
        '''
        d_real, _ = netDis(x_real)
        d_label_real = torch.ones(batch_size, 1, device=device)

        d_real_loss = nn.BCELoss()(d_real, d_label_real)

        '''
        TODO: Compute probability of fake data and construct labels
        d_fake = 
        d_label = 
        '''
        z = torch.randn(batch_size, z_dim, device=device)
        x_fake = netGen(z)
        d_fake, _ = netDis(x_fake.detach())
        d_label_fake = torch.zeros(batch_size, 1, device=device)

        d_fake_loss = nn.BCELoss()(d_fake, d_label_fake)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optDis.step()

        # Train Generator
        netGen.zero_grad()

        '''
        TODO: Compute probability of fake data and construct labels
        d_label = 
        d_fake = 
        '''
        d_fake, _ = netDis(x_fake)
        d_label_real = torch.ones(batch_size, 1, device=device)  # Generator wants discriminator to believe fake is real

        g_loss = nn.BCELoss()(d_fake, d_label_real)
        g_loss.backward()
        optGen.step()

        # Show Information
        if i % 100 == 0:
            print("[%d/%d][%s/%d] D_loss: %.4f | G_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), d_loss.item(), g_loss.item()))
        
        if i % 500 == 0:
            print("Generate Images & Save Models ...")
            x_fixed = netGen(z_fixed).cpu().detach()
            plt.figure(figsize=(6, 6))
            plt.imshow(np.transpose(vutils.make_grid(x_fixed, nrow=6, padding=2, normalize=True).cpu(), (1, 2, 0)))
            plt.axis("off")
            plt.savefig(out_folder + f"{str(epoch).zfill(2)}_{str(i).zfill(4)}.jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netGen.state_dict(), save_folder + "netGen.pt")
            torch.save(netDis.state_dict(), save_folder + "netDis.pt")
