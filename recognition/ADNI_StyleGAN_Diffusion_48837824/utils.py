import torch
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import numpy as np

from modules import ConditionalMappingNetwork as MappingNetwork

"""
utils.py 
Utility functions for Conditional StyleGAN2 training on AD vs NC MRI data.
Author: Tyreece Paul
"""

# Training Configuration
DATASET = "ADNI/AD_NC/train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 150  

# Learning Configuration
LEARNING_RATE = 0.001  # Reduced from 0.002 - matches proven working config
BATCH_SIZE = 4  # Balanced for 256×256 on RTX 4070 (32 was too large, causes OOM)

# Architecture Configuration
LOG_RESOLUTION = 8  
Z_DIM = 256  # Reduced from 512 - matches working config
W_DIM = 256  # Reduced from 512 - matches working config

# Regularization (Official StyleGAN2 approach)
# Lower R1 gamma prevents discriminator from overpowering generator
# Official StyleGAN2 uses 1.0-10.0 depending on dataset difficulty
R1_GAMMA = 10.0  # WGAN-GP style regularization (lambda_gp = 10)
PL_WEIGHT = 2.0  # Higher path length penalty reduces checkerboard artifacts  

# Data Loading
NUM_WORKERS = 8  # Increased from 4 for faster data loading
RANDOM_SEED = 42
VAL_SPLIT = 0.1

# Training Intervals
SAVE_INTERVAL = 25
VAL_INTERVAL = 1  
VAL_SAMPLES = 8  

CLASS_NAMES = ['AD', 'NC']  

# Performance
USE_AMP = True 

# Initialize mapping network
mapping_network = MappingNetwork(Z_DIM, W_DIM, num_layers=8).to(DEVICE)

def get_w(batch_size, mapping_network, labels, device=DEVICE):
    """  
    Generate w latent vectors from random z noise for conditional generation.
    Args:
        batch_size (int): Number of samples to generate.
        mapping_network (nn.Module): The mapping network to convert z to w.
        labels (torch.Tensor): Class labels for conditional generation.
        device (str): Device to perform computation on.
    Returns:
        torch.Tensor: Latent vectors in w space, shape (batch_size, num_layers, W_DIM).
    """

    z = torch.randn(batch_size, Z_DIM, device=device)
    w = mapping_network(z, labels)  
    
    # Calculate number of style injection points using Generator formula
    # Generator expects: 2 * (log_res - 1) layers
    num_layers = 2 * (LOG_RESOLUTION - 1)  # e.g., 2 * (8-1) = 14
    
    # Broadcast w to all layers
    w = w.unsqueeze(1).expand(-1, num_layers, -1)
    return w

def get_noise(batch_size, device=DEVICE):
    """
    Generate noise inputs for the generator at each resolution level.
    Args:
        batch_size (int): Number of samples to generate.
        device (str): Device to perform computation on.
    Returns:
        list of torch.Tensor: List of noise tensors for each resolution level.
    """

    noise_list = []
    resolution = 4
    
    # Initial block (4x4) - single noise
    noise_list.append((torch.randn(batch_size, 1, resolution, resolution, device=device),))
    
    # Upsampling blocks - two noise inputs each
    for _ in range(LOG_RESOLUTION - 2):
        resolution *= 2
        n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        noise_list.append((n1, n2))
    
    return noise_list


def generate_examples(gen, mapping_network, epoch, n=16, device='cuda'):
    """
    Generate and save example images from the generator.
    Applies multiple visualization techniques to ensure visibility of outputs.
    Args:
        gen (nn.Module): The generator model.
        mapping_network (nn.Module): The mapping network.
        epoch (int): Current epoch number (for saving files).
        n (int): Number of images to generate.
        device (str): Device to perform computation on.
    """

    gen.eval()
    mapping_network.eval()
    
    folder = 'saved_examples'
    os.makedirs(folder, exist_ok=True)
    
    with torch.no_grad():
        from utils import get_w, get_noise
        
        # Generate random class labels for mixed examples
        random_labels = torch.randint(0, 2, (n,), device=device)
        w = get_w(n, mapping_network, random_labels, device=device)
        noise = get_noise(n, device=device)
        fake_imgs = gen(w, noise)
        
        # Get statistics
        img_min = fake_imgs.min().item()
        img_max = fake_imgs.max().item()
        img_mean = fake_imgs.mean().item()
        img_std = fake_imgs.std().item()
        
        print(f"\n[Epoch {epoch}] Generated {n} images:")
        print(f"  Range: [{img_min:.3f}, {img_max:.3f}] (span: {img_max - img_min:.3f})")
        print(f"  Mean: {img_mean:.3f}, Std: {img_std:.3f}")
        
        # Method 1: Standard denormalization (for well-trained models)
        imgs_standard = fake_imgs * 0.5 + 0.5
        imgs_standard = torch.clamp(imgs_standard, 0, 1)
        
        save_image(imgs_standard, f"{folder}/epoch{epoch}_standard.png", 
                   nrow=4, padding=2, normalize=False)
        
        # Method 2: Adaptive contrast stretching (for early training)
        # This makes low-variance outputs visible
        if img_std < 0.3:  # Low variance - apply stretching
            imgs_stretched = (fake_imgs - img_min) / (img_max - img_min + 1e-8)
            imgs_stretched = torch.clamp(imgs_stretched, 0, 1)
            
            save_image(imgs_stretched, f"{folder}/epoch{epoch}_stretched.png",
                       nrow=4, padding=2, normalize=False)
            print(f"Applied contrast stretching (low variance detected)")
        else:
            imgs_stretched = imgs_standard
        
        # Method 3: Histogram equalization for visibility
        imgs_eq = (fake_imgs - fake_imgs.mean()) / (fake_imgs.std() + 1e-8)
        imgs_eq = imgs_eq * 0.2 + 0.5  # Scale to visible range
        imgs_eq = torch.clamp(imgs_eq, 0, 1)
        
        save_image(imgs_eq, f"{folder}/epoch{epoch}_equalized.png",
                   nrow=4, padding=2, normalize=False)
        
        # Create comparison grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Show first 4 images with different processing
        for idx in range(min(4, n)):
            row, col = idx // 2, idx % 2
            
            # Use stretched version for better visibility
            img = imgs_stretched[idx].cpu().permute(1, 2, 0).numpy()
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Sample {idx+1}\n'
                                     f'Range: [{fake_imgs[idx].min():.2f}, '
                                     f'{fake_imgs[idx].max():.2f}]')
            axes[row, col].axis('off')
        
        plt.suptitle(f'Epoch {epoch} - Generated Samples\n'
                     f'Mean: {img_mean:.3f}, Std: {img_std:.3f}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{folder}/epoch{epoch}_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save statistics plot
        if epoch > 0:
            update_stats_plot(epoch, img_min, img_max, img_mean, img_std, folder)
        
        print(f"Saved standard: epoch{epoch}_standard.png")
        print(f"Saved stretched: epoch{epoch}_stretched.png")
        print(f"Saved analysis: epoch{epoch}_analysis.png")
        
    gen.train()
    mapping_network.train()


def update_stats_plot(epoch, img_min, img_max, img_mean, img_std, folder):
    """
    Track generator output statistics over training.
    Saves a plot of min, max, mean, std over epochs.
    Args:
        epoch (int): Current epoch number.
        img_min (float): Minimum pixel value of generated images.
        img_max (float): Maximum pixel value of generated images.
        img_mean (float): Mean pixel value of generated images.
        img_std (float): Standard deviation of pixel values.
        folder (str): Folder to save the stats file and plot.
    """
    
    stats_file = f"{folder}/generation_stats.txt"
    
    # Append stats
    with open(stats_file, 'a') as f:
        f.write(f"{epoch},{img_min},{img_max},{img_mean},{img_std}\n")
    
    # Read all stats
    try:
        data = np.loadtxt(stats_file, delimiter=',')
        if len(data.shape) == 1:  # Only one epoch
            data = data.reshape(1, -1)
        
        epochs = data[:, 0]
        mins = data[:, 1]
        maxs = data[:, 2]
        means = data[:, 3]
        stds = data[:, 4]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Min/Max range
        axes[0, 0].plot(epochs, mins, 'b-', label='Min', linewidth=2)
        axes[0, 0].plot(epochs, maxs, 'r-', label='Max', linewidth=2)
        axes[0, 0].axhline(y=-1, color='k', linestyle='--', alpha=0.3, label='Tanh limits')
        axes[0, 0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].set_title('Generator Output Range')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Mean
        axes[0, 1].plot(epochs, means, 'g-', linewidth=2)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Target mean')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Mean Value')
        axes[0, 1].set_title('Generator Output Mean')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: Std deviation
        axes[1, 0].plot(epochs, stds, 'purple', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Std Deviation')
        axes[1, 0].set_title('Generator Output Variance')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Range span
        spans = maxs - mins
        axes[1, 1].plot(epochs, spans, 'orange', linewidth=2)
        axes[1, 1].axhline(y=2.0, color='k', linestyle='--', alpha=0.3, 
                          label='Full tanh range')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Range Span')
        axes[1, 1].set_title('Generator Dynamic Range')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle('Generator Output Statistics Over Training', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{folder}/training_statistics.png", dpi=150)
        plt.close()
        
    except Exception as e:
        pass  # First epoch, no history yet


def generate_interpolation(gen, mapping_network, epoch, device='cuda'):
    """
    Generate and save interpolation images between AD and NC classes.
    Args:
        gen (nn.Module): The generator model.
        mapping_network (nn.Module): The mapping network.
        epoch (int): Current epoch number (for saving files).
        device (str): Device to perform computation on.
    """

    gen.eval()
    mapping_network.eval()
    
    folder = 'saved_examples'
    os.makedirs(folder, exist_ok=True)
    
    with torch.no_grad():
        # Fixed random seed for consistent comparison
        z = torch.randn(1, Z_DIM, device=device)
        noise = get_noise(1, device)
        
        # Generate from both classes with same z
        imgs = []
        for class_idx in [0, 1]:
            class_labels = torch.tensor([class_idx], dtype=torch.long, device=device)
            w = get_w(1, mapping_network, class_labels, device)
            img = gen(w, noise)
            imgs.append(img)
        
        # Interpolate between AD and NC
        num_steps = 8
        interpolated = []
        for alpha in torch.linspace(0, 1, num_steps):
            # Interpolate in w space
            class_labels_ad = torch.tensor([0], dtype=torch.long, device=device)
            class_labels_nc = torch.tensor([1], dtype=torch.long, device=device)
            
            w_ad = get_w(1, mapping_network, class_labels_ad, device)
            w_nc = get_w(1, mapping_network, class_labels_nc, device)
            
            w_interp = (1 - alpha) * w_ad + alpha * w_nc
            img_interp = gen(w_interp, noise)
            img_interp = img_interp * 0.5 + 0.5
            interpolated.append(img_interp)
        
        interpolated = torch.cat(interpolated, dim=0)
        save_image(interpolated, f"{folder}/epoch{epoch}_interpolation.png",
                  nrow=num_steps, padding=2, normalize=False)
        
        print(f"  ✓ Saved interpolation: epoch{epoch}_interpolation.png")
    
    gen.train()
    mapping_network.train()


def calculate_gradient_penalty_r1(disc, real_imgs, device=DEVICE):
    """
    Calculate R1 gradient penalty for the discriminator.
    Args:
        disc (nn.Module): The discriminator model.
        real_imgs (torch.Tensor): Real images from the dataset.
        device (str): Device to perform computation on.
    Returns:
        torch.Tensor: R1 gradient penalty.
    """
    
    real_imgs.requires_grad_(True)
    real_pred = disc(real_imgs)
    
    gradients = torch.autograd.grad(
        outputs=real_pred.sum(),
        inputs=real_imgs,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # R1 penalty: E[||∇D(x)||^2]
    r1_penalty = gradients.pow(2).reshape(real_imgs.size(0), -1).sum(1).mean()
    
    return r1_penalty
