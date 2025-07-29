import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from read_celeba import *

# Decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Construct Model
z_dim = 128
netEnc = models.Encoder(z_dim).to(device)
netDec = models.Generator(z_dim).to(device)
params = list(netEnc.parameters()) + list(netDec.parameters())
opt = optim.Adam(params, lr=2e-4, betas=(0.5, 0.999))

# Create Results Folder
model_name = "vae"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder =  "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Main Loop
num_epochs = 20
z_fixed = torch.randn(36, z_dim, device=device)
print("Start Training ...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Initialize
        netEnc.zero_grad()
        netDec.zero_grad()

        # Forward 
        '''
        TODO: Forward Computation of the Network
        1. Compute z_mean and z_logvar.
        2. Sample z from z_mean and z_logvar.
        3. Compute reconstruction of x.
        '''
        x = data[0].to(device)                                 # Input batch
        z_mu, z_logvar = netEnc(x)                             # Step 1
        std = torch.exp(0.5 * z_logvar)                        # Reparameterization trick
        eps = torch.randn_like(std)
        z = z_mu + eps * std                                   # Step 2
        x_recon = netDec(z)                                    # Step 3

        # Loss and Optimization
        '''
        TODO: Loss Computation
        rec_loss = 
        kl_loss = 
        '''
        rec_loss = F.mse_loss(x_recon, x)                      # Reconstruction loss
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp()) / x.size(0)  # KL Divergence
        loss = rec_loss + 0.0001 * kl_loss
        loss.backward()
        opt.step()

        # Show Information
        if i % 100 == 0:
            print("[%d/%d][%s/%d] R_loss: %.4f | KL_loss: %.4f"\
            %(epoch+1, num_epochs, str(i).zfill(4), len(dataloader), rec_loss.item(), kl_loss.item()))
        
        if i % 500 == 0:
            print("Generate Images & Save Models ...")
            # Output Images
            x_fixed = netDec(z_fixed).cpu().detach()
            plt.figure(figsize=(6,6))
            plt.imshow(np.transpose(vutils.make_grid(x_fixed, nrow=6, padding=2, normalize=True).cpu(),(1,2,0)))
            plt.axis("off")
            plt.savefig(out_folder+str(epoch).zfill(2)+"_"+str(i).zfill(4)+".jpg", bbox_inches="tight")
            plt.close()
            # Save Model
            torch.save(netEnc.state_dict(), save_folder+"netEnc.pt")
            torch.save(netDec.state_dict(), save_folder+"netDec.pt")
