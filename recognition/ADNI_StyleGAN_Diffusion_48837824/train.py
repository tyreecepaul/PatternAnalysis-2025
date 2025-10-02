import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules import Generator, Discriminator, PathLengthPenalty
from dataset import get_dataloaders

from utils import gradient_penalty, get_w, get_noise, generate_examples
from utils import DEVICE, EPOCHS, LEARNING_RATE, BATCH_SIZE, LOG_RESOLUTION, CRITIC_ITER, SAVE_INTERVAL, VAL_INTERVAL, USE_AMP, PL_WEIGHT, W_DIM, LAMBDA_GP, VAL_SAMPLES, mapping_network

"""
train.py
Training loop for StyleGAN2
"""

gen = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
disc = Discriminator(LOG_RESOLUTION).to(DEVICE)
pl_penalty = PathLengthPenalty().to(DEVICE)

# Optimizers
opt_gen = optim.Adam(list(gen.parameters()) + list(mapping_network.parameters()), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

# Loss placeholder (WGAN-GP uses critic scores)
loss_fn = nn.MSELoss()

d_losses = []
g_losses = []
val_d_scores = []

train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

scaler_gen = torch.amp.GradScaler(enabled=USE_AMP)
scaler_disc = torch.amp.GradScaler(enabled=USE_AMP)

for epoch in range(1, EPOCHS + 1):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    running_d_loss = 0.0
    running_g_loss = 0.0

    for real_imgs, _ in pbar:
        real_imgs = real_imgs.to(DEVICE)
        batch_size = real_imgs.shape[0]

        # ---------------------
        # Train Discriminator
        # ---------------------
        for _ in range(CRITIC_ITER):
            disc.zero_grad()
            w = get_w(batch_size, mapping_network)
            noise = get_noise(batch_size)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                fake_imgs = gen(w, noise).detach()
                disc_real = disc(real_imgs)
                disc_fake = disc(fake_imgs)
                gp = gradient_penalty(disc, real_imgs, fake_imgs, device=DEVICE)
                d_loss = disc_fake.mean() - disc_real.mean() + LAMBDA_GP * gp

            scaler_disc.scale(d_loss).backward()
            scaler_disc.step(opt_disc)
            scaler_disc.update()

        # ---------------------
        # Train Generator
        # ---------------------
        gen.zero_grad()
        mapping_network.zero_grad()
        w = get_w(batch_size, mapping_network)
        noise = get_noise(batch_size)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            fake_imgs = gen(w, noise)
            disc_fake = disc(fake_imgs)
            g_loss = -disc_fake.mean()
            pl_loss = pl_penalty(fake_imgs, w)
            total_g_loss = g_loss + PL_WEIGHT * pl_loss

        scaler_gen.scale(total_g_loss).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        running_d_loss += d_loss.item()
        running_g_loss += total_g_loss.item()

        pbar.set_postfix({
            "D_loss": d_loss.item(),
            "G_loss": total_g_loss.item(),
            "PL_loss": pl_loss.item()
        })

    # ---------------------
    # Average losses for epoch
    # ---------------------
    avg_d_loss = running_d_loss / len(train_loader)
    avg_g_loss = running_g_loss / len(train_loader)
    d_losses.append(avg_d_loss)
    g_losses.append(avg_g_loss)

    # ---------------------
    # Save generated examples
    # ---------------------
    if epoch % SAVE_INTERVAL == 0:
        generate_examples(gen, mapping_network, epoch, n=VAL_SAMPLES)

    # ---------------------
    # Validation evaluation
    # ---------------------
    if epoch % VAL_INTERVAL == 0:
        gen.eval()
        disc.eval()
        with torch.no_grad():
            val_scores = []
            for val_imgs, _ in val_loader:
                val_imgs = val_imgs.to(DEVICE)
                val_scores.append(disc(val_imgs).mean().item())
            avg_val_score = sum(val_scores) / len(val_scores)
            val_d_scores.append(avg_val_score)
            print(f"Validation Discriminator Score: {avg_val_score:.4f}")
        gen.train()
        disc.train()

    # ---------------------
    # Save checkpoints
    # ---------------------
    if epoch % 50 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(gen.state_dict(), f"checkpoints/gen_epoch{epoch}.pth")
        torch.save(disc.state_dict(), f"checkpoints/disc_epoch{epoch}.pth")
        torch.save(mapping_network.state_dict(), f"checkpoints/mapping_epoch{epoch}.pth")

# -------------------------
# Plot loss curves
# -------------------------
plt.figure(figsize=(8,6))
plt.plot(d_losses, label="Discriminator Loss")
plt.plot(g_losses, label="Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GAN Losses Over Epochs")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

# Plot validation discriminator score
plt.figure(figsize=(8,6))
plt.plot(range(VAL_INTERVAL, EPOCHS+1, VAL_INTERVAL), val_d_scores, label="Validation Discriminator Score")
plt.xlabel("Epoch")
plt.ylabel("Average Score")
plt.title("Validation Discriminator Scores")
plt.legend()
plt.grid(True)
plt.savefig("val_score_curve.png")
plt.show()

print("Training complete!")