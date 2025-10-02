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
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
LOG_RESOLUTION = 7 #2^7: 128*128
Z_DIM = 256
W_DIM = 256
LAMBDA_GP = 10

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
    Returns shape [LOG_RESOLUTION, batch_size, W_DIM]
    """
    z = torch.randn(batch_size, Z_DIM, device=device)
    w = mapping_network(z)  # [batch_size, W_DIM]
    # Expand along style layers for each resolution
    w = w.unsqueeze(0).expand(LOG_RESOLUTION, -1, -1)
    return w


def get_noise(batch_size, device=DEVICE):
    """
    Returns list of tuples (noise1, noise2) per resolution for StyleGAN2 generator.
    noise1 can be None for the first block.
    """
    noise_list = []
    resolution = 4

    for i in range(LOG_RESOLUTION):
        n1 = None if i == 0 else torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        noise_list.append((n1, n2))
        resolution *= 2

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

