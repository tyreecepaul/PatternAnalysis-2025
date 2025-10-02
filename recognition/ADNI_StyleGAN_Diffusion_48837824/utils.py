import torch
from torchvision.utils import save_image
import os

from modules import MappingNetwork

"""
utils.py
Utility functions for training and using StyleGAN2
"""

# Hyperparameters
DATASET = "ADNI/AD_NC/train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 8  # Reduced from 32 to avoid GPU memory issues
LOG_RESOLUTION = 8 #2^8: 256*256, but we'll use 2^7 (128x128)
Z_DIM = 512  # Latent dimension for input noise
W_DIM = 512  # Learned intermediate latent dimension - must be >= max channels
LAMBDA_GP = 10
NUM_WORKERS = 4
RANDOM_SEED = 42
VAL_SPLIT = 0.1 
PL_WEIGHT = 2.0
CRITIC_ITER = 1
SAVE_INTERVAL = 10
VAL_INTERVAL = 10
USE_AMP = True
VAL_SAMPLES = 16

# Create mapping network with correct dimensions
mapping_network = MappingNetwork(Z_DIM, W_DIM).to(DEVICE)

def gradient_penalty(critic, real, fake, device=DEVICE):
    batch_size, C, H, W = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1), device=device).expand_as(real)
    interpolated = alpha * real + (1 - alpha) * fake.detach()
    interpolated.requires_grad_(True)

    mixed_scores = critic(interpolated)

    grad_outputs = torch.ones_like(mixed_scores, device=device)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp


def get_w(batch_size, mapping_network, device=DEVICE):
    """
    Sample z ~ N(0,1) and map to w using the mapping network.
    Returns shape [batch_size, num_style_layers, W_DIM] for StyleGAN2 generator
    """
    z = torch.randn(batch_size, Z_DIM, device=device)
    w = mapping_network(z)  # [batch_size, W_DIM]
    # StyleGAN2 needs: 1 initial layer + 2 per upsampling block
    # For LOG_RESOLUTION=8: initial (4x4) + 6 blocks = 1 + 2*6 = 13 style layers total
    num_blocks = len([min(512, 64 * (2 ** i)) for i in range(LOG_RESOLUTION - 2, -1, -1)]) - 1  # number of upsampling blocks
    num_style_layers = 1 + 2 * num_blocks  # initial + 2 per block
    w = w.unsqueeze(1).expand(-1, num_style_layers, -1)  # [batch_size, num_style_layers, W_DIM]
    return w


def get_noise(batch_size, device=DEVICE):
    """
    Returns list of tuples (noise1, noise2) per resolution for StyleGAN2 generator.
    Initial resolution is 4x4, then doubles each time.
    """
    noise_list = []
    resolution = 4

    # First noise for initial 4x4 resolution  
    n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
    noise_list.append((n1,))  # Single noise for initial block
    
    # Noise for upsampling blocks
    # Calculate number of blocks based on actual generator architecture
    channels = [min(512, 64 * (2 ** i)) for i in range(LOG_RESOLUTION - 2, -1, -1)]
    num_blocks = len(channels) - 1  # number of upsampling blocks
    
    for i in range(num_blocks):
        resolution *= 2
        n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        noise_list.append((n1, n2))

    return noise_list


def generate_examples(gen, mapping_network, epoch, n=100, device=DEVICE):
    """
    Generate 'n' images using the generator and save to saved_examples/epoch{epoch}/
    """
    gen.eval()
    for i in range(n):
        with torch.no_grad():
            w = get_w(1, mapping_network, device=device)
            noise = get_noise(1, device=device)
            img = gen(w, noise)
            folder = f'saved_examples/epoch{epoch}'
            os.makedirs(folder, exist_ok=True)
            save_image(img * 0.5 + 0.5, f"{folder}/img_{i}.png")
    gen.train()

