import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# PyTorch optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from modules import Generator, PathLengthPenalty, EMA
from modules import ConditionalDiscriminator as Discriminator
from dataset import get_dataloaders
from utils import (
    get_w, get_noise, generate_examples, generate_interpolation,
    DEVICE, LEARNING_RATE, BATCH_SIZE, LOG_RESOLUTION, USE_AMP, 
    W_DIM, Z_DIM, VAL_SAMPLES, mapping_network, EPOCHS, SAVE_INTERVAL, 
    VAL_INTERVAL, PL_WEIGHT, R1_GAMMA, CLASS_NAMES
)

"""
Conditional StyleGAN2 training script for AD vs NC.

This script runs class-conditional StyleGAN2 training using a projection-based
discriminator and a conditional mapping network. It implements the common
StyleGAN2 training recipe with the following features:

- Mixed precision (AMP) for memory and performance.
- Lazy R1 regularization for the discriminator and path-length regularization for the generator.
- Projection-based conditioning (class embeddings used in discriminator projection term).
- Checkpointing and sample generation (saved to `checkpoints/` and `saved_examples/`).

Usage:
    python train.py

Hyperparameters and dataset paths are controlled from `utils.py`. The script
prints training progress and saves periodic checkpoints and sample images.

Outputs:
- Checkpoints: ./checkpoints/conditional_stylegan2_epoch{epoch}.pth
- Sample images and analysis: saved_examples/
- Final training summary plot: conditional_training_summary.png
"""

# Training Configuration
D_REG_INTERVAL = 16  # Apply R1 regularization every 16 discriminator updates
G_REG_INTERVAL = 8   # Apply path length regularization every 8 generator updates (reduced frequency)

# Balanced Learning Rates (Official StyleGAN2 approach)
# Fix discriminator overpowering: D much slower, G:D ratio should be ~10:1
D_LEARNING_RATE = LEARNING_RATE * 0.1   # 0.0002 - much lower for D
G_LEARNING_RATE = LEARNING_RATE          # 0.002 - keep for G

# Official StyleGAN2 uses 1:1 update ratio, not multiple G updates
# Balance is achieved through LR and regularization, not update frequency

print(f"Using device: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Batch size: {BATCH_SIZE}")
print(f"Resolution: {2**LOG_RESOLUTION}x{2**LOG_RESOLUTION}")
print(f"Training epochs: {EPOCHS}")
print(f"Classes: {CLASS_NAMES}")
print("-" * 60)

# Initialize models (Generator stays the same, Discriminator is conditional)
# Note: mapping_network is defined and instantiated in `utils.mapping_network`.
# `mapping_network` parameters passed into the generator optimizer so the
# mapping network is trained jointly with the generator.
gen = Generator(LOG_RESOLUTION, W_DIM).to(DEVICE)
disc = Discriminator(LOG_RESOLUTION, num_classes=2).to(DEVICE)
pl_penalty = PathLengthPenalty().to(DEVICE)

# EMA Generator (Official StyleGAN2 feature for better evaluation)
gen_ema = EMA(gen, decay=0.999)

# Get the number of layers from the generator
NUM_LAYERS = gen.num_layers

# Helper function to get w with correct number of layers
def get_w_correct(batch_size, mapping_network, labels, num_layers=NUM_LAYERS, device=DEVICE):
    """Generate w latent vectors with correct number of layers for this generator."""
    z = torch.randn(batch_size, W_DIM, device=device)
    w = mapping_network(z, labels)
    # Broadcast w to all layers using the generator's num_layers
    w = w.unsqueeze(1).expand(-1, num_layers, -1)
    return w

# Optimizers
opt_gen = optim.Adam(
    list(gen.parameters()) + list(mapping_network.parameters()),
    lr=G_LEARNING_RATE,
    betas=(0.0, 0.99),
    eps=1e-8
)
opt_disc = optim.Adam(
    disc.parameters(),
    lr=D_LEARNING_RATE,
    betas=(0.0, 0.99),
    eps=1e-8
)

# Loss tracking
d_losses = []
g_losses = []
r1_losses = []
pl_losses = []

# Track per-class statistics
ad_count = 0
nc_count = 0

train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

# AMP scalers
scaler_gen = torch.amp.GradScaler('cuda', enabled=USE_AMP)
scaler_disc = torch.amp.GradScaler('cuda', enabled=USE_AMP)

print("=" * 60)
print("STARTING CONDITIONAL STYLEGAN2 TRAINING")
print(f"  • Class-conditional generation (AD vs NC)")
print(f"  • Projection-based conditioning")
print(f"  • Non-saturating loss with R1 regularization")
print(f"  • R1 gamma: {R1_GAMMA}, applied every {D_REG_INTERVAL} steps")
print(f"  • Path length weight: {PL_WEIGHT}, applied every {G_REG_INTERVAL} steps")
print(f"  • Generator LR: {G_LEARNING_RATE}, Discriminator LR: {D_LEARNING_RATE}")
print(f"  • Official StyleGAN2: 1:1 G:D update ratio")
print("=" * 60)

global_step = 0

for epoch in range(1, EPOCHS + 1):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    epoch_r1 = 0.0
    epoch_pl = 0.0
    
    epoch_ad_count = 0
    epoch_nc_count = 0

    for batch_idx, (real_imgs, labels) in enumerate(pbar):
        global_step += 1
        real_imgs = real_imgs.to(DEVICE)
        labels = labels.to(DEVICE)  # Class labels: 0=AD, 1=NC
        batch_size = real_imgs.shape[0]
        
        # Track class distribution
        epoch_ad_count += (labels == 0).sum().item()
        epoch_nc_count += (labels == 1).sum().item()

        # Train Discriminator 
        disc.zero_grad()
        
        # Generate fake images with SAME class distribution as real batch
        # Apply style mixing (official StyleGAN2 feature for better disentanglement)
        if torch.rand(()).item() < 0.9:  # 90% of the time use style mixing
            # Generate two different w vectors
            w1 = get_w_correct(batch_size, mapping_network, labels)
            w2 = get_w_correct(batch_size, mapping_network, labels)
            # Random crossover point
            crossover = torch.randint(1, NUM_LAYERS, ()).item()
            # Mix styles: use w1 for lower layers, w2 for higher layers
            w = w1.clone()
            w[:, crossover:] = w2[:, crossover:]
        else:
            # Regular single style
            w = get_w_correct(batch_size, mapping_network, labels)
        
        noise = get_noise(batch_size)
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            fake_imgs = gen(w, noise).detach()
            
            # Conditional discriminator - pass class labels
            real_pred = disc(real_imgs, labels)
            fake_pred = disc(fake_imgs, labels)
            
            # Non-saturating loss
            d_loss = torch.nn.functional.softplus(fake_pred).mean()
            d_loss = d_loss + torch.nn.functional.softplus(-real_pred).mean()
        
        scaler_disc.scale(d_loss).backward()
        
        # R1 Regularization (lazy - only every D_REG_INTERVAL steps)
        r1_loss = torch.tensor(0.0, device=DEVICE)
        if global_step % D_REG_INTERVAL == 0:
            real_imgs.requires_grad_(True)
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                real_pred = disc(real_imgs, labels)
            
            # Compute R1 gradient penalty
            r1_grads = torch.autograd.grad(
                outputs=real_pred.sum(),
                inputs=real_imgs,
                create_graph=True
            )[0]
            
            r1_penalty = r1_grads.pow(2).reshape(batch_size, -1).sum(1).mean()
            r1_loss = r1_penalty * (R1_GAMMA / 2)
            
            # Scale by interval for proper learning rate
            weighted_r1 = r1_loss * D_REG_INTERVAL
            
            scaler_disc.scale(weighted_r1).backward()
            
            epoch_r1 += r1_loss.item()
        
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        # Train Generator
        gen.zero_grad()
        mapping_network.zero_grad()
        
        w = get_w_correct(batch_size, mapping_network, labels)
        noise = get_noise(batch_size)
        
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            fake_imgs = gen(w, noise)
            fake_pred = disc(fake_imgs, labels)
            
            # Non-saturating generator loss
            g_loss = torch.nn.functional.softplus(-fake_pred).mean()
        
        scaler_gen.scale(g_loss).backward()
        
        # Path Length Regularization (lazy)
        pl_loss = torch.tensor(0.0, device=DEVICE)
        if global_step % G_REG_INTERVAL == 0:
            # Generate new samples for path length
            w_pl = get_w_correct(batch_size, mapping_network, labels)
            w_pl.requires_grad_(True)
            noise_pl = get_noise(batch_size)
            
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                fake_imgs_pl = gen(w_pl, noise_pl)
                pl_loss = pl_penalty(fake_imgs_pl, w_pl)
            
            # Scale by interval
            weighted_pl = pl_loss * PL_WEIGHT * G_REG_INTERVAL
            
            scaler_gen.scale(weighted_pl).backward()
            
            epoch_pl += pl_loss.item()
        
        scaler_gen.step(opt_gen)
        scaler_gen.update()
        
        # Update EMA generator 
        gen_ema.update()

        # Logging
        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        
        pbar.set_postfix({
            'D': f'{d_loss.item():.3f}',
            'G': f'{g_loss.item():.3f}',
            'R1': f'{r1_loss.item():.3f}',
            'PL': f'{pl_loss.item():.3f}',
            'AD': epoch_ad_count,
            'NC': epoch_nc_count
        })

    # Epoch summary
    avg_d = epoch_d_loss / len(train_loader)
    avg_g = epoch_g_loss / len(train_loader)
    avg_r1 = epoch_r1 / max(1, len(train_loader) // D_REG_INTERVAL)
    avg_pl = epoch_pl / max(1, len(train_loader) // G_REG_INTERVAL)
    
    d_losses.append(avg_d)
    g_losses.append(avg_g)
    r1_losses.append(avg_r1)
    pl_losses.append(avg_pl)
    
    print(f"\nEpoch {epoch} Summary:")
    print(f"  D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")
    print(f"  R1: {avg_r1:.4f} | PL: {avg_pl:.4f}")
    print(f"  Class distribution - AD: {epoch_ad_count}, NC: {epoch_nc_count}")

    # Generate samples from BOTH classes (using EMA generator for better quality)
    if epoch % VAL_INTERVAL == 0:
        # Use EMA generator for evaluation 
        gen_ema.apply_shadow()
        generate_examples(gen, mapping_network, epoch, n=VAL_SAMPLES)
        
        # Generate interpolation between classes
        if epoch % (SAVE_INTERVAL * 2) == 0:
            generate_interpolation(gen, mapping_network, epoch)
        
        # Restore training generator
        gen_ema.restore()
    
    # Save checkpoints
    if epoch % SAVE_INTERVAL == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'gen_state': gen.state_dict(),
            'gen_ema_shadow': gen_ema.shadow,  # Save EMA parameters
            'disc_state': disc.state_dict(),
            'mapping_state': mapping_network.state_dict(),
            'opt_gen': opt_gen.state_dict(),
            'opt_disc': opt_disc.state_dict(),
            'class_names': CLASS_NAMES,
        }, f"checkpoints/conditional_stylegan2_epoch{epoch}.pth")
        # The checkpoint includes optimizer states and EMA so training can be resumed
        # (useful if training is interrupted). Intentionally save every 25
        # epochs to balance disk usage and recovery granularity.
        print(f"  ✓ Saved checkpoint: conditional_stylegan2_epoch{epoch}.pth")

# Final plots
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(d_losses, label='D Loss', linewidth=2, color='blue')
plt.plot(g_losses, label='G Loss', linewidth=2, color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Losses')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(r1_losses, label='R1 Penalty', color='orange', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('R1 Loss')
plt.title('R1 Gradient Penalty')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(pl_losses, label='Path Length', color='green', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('PL Loss')
plt.title('Path Length Regularization')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("conditional_training_summary.png", dpi=200)

print(f"\n{'='*60}")
print("CONDITIONAL TRAINING COMPLETED!")
print(f"{'='*60}")
print(f"Total AD samples seen: {ad_count}")
print(f"Total NC samples seen: {nc_count}")
print("\nYou can now generate:")
print("  • AD (Alzheimer's) images by passing class_label=0")
print("  • NC (Normal Control) images by passing class_label=1")
print(f"{'='*60}")