""" 
modules.py
Conditional StyleGAN2 Model Implementation for ADNI Dataset
- Model architecture adapted for conditional generation (AD vs NC)
- Includes conditional mapping network and projection-based discriminator
- Implementation based on the original StyleGAN2 paper and PyTorch framework
"""

import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

class MappingNetwork(nn.Module):
    """
    Mapping network for StyleGAN/StyleGAN2.

    This network maps the initial **latent vector z** (sampled from a
    standard normal distribution) to an intermediate **latent vector w**.
    The 'w' vector is considered more disentangled and is used to drive
    the style modulation throughout the synthesis network.
    """

    def __init__(self, z_dim: int, w_dim: int, num_layers: int = 8):
        """
        Initializes the MappingNetwork.

        The network consists of a PixelNorm operation followed by a stack of
        fully-connected layers with LeakyReLU activations and equalized learning rates.

        Args:
            z_dim (int): Dimension of the input latent vector $\mathbf{z}$.
            w_dim (int): Dimension of the output intermediate latent vector $\mathbf{w}$.
            num_layers (int, optional): The number of fully-connected layers
                in the mapping network. Defaults to 8.
        """
        
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            # Use EqualizedLinear for stabilized training (StyleGAN concept)
            layers.append(EqualizedLinear(in_dim, w_dim))
            # Use LeakyReLU with a negative slope of 0.2
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        self.mapping = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the mapping network.

        The process involves:
        1. **PixelNorm**: Normalizing the input $\mathbf{z}$ to a unit length.
        2. **Sequential Mapping**: Passing the normalized vector through the 
           stack of linear layers and non-linearities.

        Args:
            z (torch.Tensor): Input latent vector $\mathbf{z}$ of shape 
                $\text{[batch\_size, z\_dim]}$.

        Returns:
            torch.Tensor: Output intermediate latent vector $\mathbf{w}$ of shape 
                $\text{[batch\_size, w\_dim]}$.
        """

        # Step 1: PixelNorm (Normalizes the input vector to unit variance)
        # Adds a small epsilon (1e-8) for numerical stability.
        x = z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)
        
        # Step 2: Sequential Mapping
        return self.mapping(x) # [batch_size, w_dim]

class ConditionalMappingNetwork(nn.Module):
    """
    Conditional mapping network that takes class labels (AD vs NC).
    """
    def __init__(self, z_dim: int, w_dim: int, num_classes: int = 2, num_layers: int = 8):
        super().__init__()
        
        # Class embedding converts 0/1 label into a learned vector
        self.class_embed = nn.Embedding(num_classes, z_dim)
        
        # First layer takes concatenated z and class embedding
        layers = []
        for i in range(num_layers):
            in_dim = z_dim * 2 if i == 0 else w_dim
            layers.append(EqualizedLinear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, z_dim]
            class_labels: [batch_size] with values 0 (AD) or 1 (NC)
        Returns:
            w: [batch_size, w_dim]
        """
        # Normalize z
        z = z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)
        
        # Get class embedding and concatenate
        class_embed = self.class_embed(class_labels)
        z_cond = torch.cat([z, class_embed], dim=1)
        
        return self.mapping(z_cond)

class GeneratorBlock(nn.Module):
    """
    A single feature resolution block in the StyleGAN2 generator.

    This block handles the generation for a specific spatial resolution 
    (e.g., from 8x8 to 16x16) and contains the core style-based modulation.
    It processes feature maps and produces a raw RGB image output for that level.
    """

    def __init__(self, W_DIM: int, in_channels: int, out_channels: int):
        """
        Initializes the GeneratorBlock.

        The block consists of two StyleBlocks (with residual connections
        handled internally by StyleBlock) and a ToRGB layer.

        Args:
            W_DIM (int): Dimension of the input style vector $\mathbf{w}$ (from the mapping network).
            in_channels (int): Number of input feature channels (from the previous block).
            out_channels (int): Number of output feature channels.
        """
        super().__init__()
        
        # StyleBlocks apply modulation, convolution, noise injection, and activation.
        self.style_block1 = StyleBlock(W_DIM, in_channels, out_channels)
        self.style_block2 = StyleBlock(W_DIM, out_channels, out_channels)
        
        # ToRGB converts feature maps to a 3-channel (RGB) image.
        self.to_rgb = ToRGB(W_DIM, out_channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the generator block.

        Args:
            x (torch.Tensor): Input feature map. 
                Shape: $\text{[batch\_size, in\_channels, H, W]}$.
            w (torch.Tensor): Style vector $\mathbf{w}$ for modulation.
                Shape: $\text{[batch\_size, W\_DIM]}$.
            noise (tuple[torch.Tensor, torch.Tensor]): A tuple containing two 
                noise tensors (one for each StyleBlock) of shape 
                $\text{[batch\_size, 1, H, W]}$.

        Returns:
            x (torch.Tensor): Output feature map after the two StyleBlocks.
                Shape: $\text{[batch\_size, out\_channels, H, W]}$.
            rgb (torch.Tensor): Raw RGB output from the ToRGB layer.
                Shape: $\text{[batch\_size, 3, H, W]}$.
        """
        # Apply the first StyleBlock
        x = self.style_block1(x, w, noise[0])
        
        # Apply the second StyleBlock (maintains the same channel count)
        x = self.style_block2(x, w, noise[1])
        
        # Generate the RGB image for this resolution level
        rgb = self.to_rgb(x, w)
        
        return x, rgb
 

class StyleBlock(nn.Module):
    """
    Style block for the StyleGAN2 generator.
    Consists of a modulated convolution, noise injection, bias addition, and activation.
    """
    def __init__(self, w_dim, in_channels, out_channels):
        """
        Style block for the StyleGAN2 generator.
        Consists of a modulated convolution, noise injection, bias addition, and activation.
        Args:
            w_dim (int): Dimension of the style vector w.
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
        """
        super().__init__()
        self.conv = Conv2dWeightModulate(in_channels, out_channels, kernel_size=3) # Modulated convolution layer
        self.to_style = EqualizedLinear(w_dim, in_channels)  # Style modulation layer, maps w to style
        self.bias = nn.Parameter(torch.zeros(out_channels)) # Bias term
        self.noise_scale = nn.Parameter(torch.zeros(1)) # Noise scale
        self.activation = nn.LeakyReLU(0.2, inplace=True) # Activation function (LeakyReLU with negative slope 0.2)

    def forward(self, x, w, noise=None):
        """
        Forward pass through the style block.
        Args:
            x (torch.Tensor): Input feature map of shape [batch_size, in_channels, H, W].
            w (torch.Tensor): Style vector of shape [batch_size, w_dim].
            noise (torch.Tensor or None): Optional noise tensor of shape [batch_size, 1, H, W] to inject.
        Returns:
            torch.Tensor: Output feature map of shape [batch_size, out_channels, H, W].
        """
        
        style = self.to_style(w) # Map w to style vector  
        x = self.conv(x, style) # Apply modulated convolution
        if noise is not None:
            x = x + self.noise_scale * noise # Inject noise
        x = x + self.bias[None, :, None, None] # Add bias
        return self.activation(x) # Apply activation [batch_size, out_channels, H, W]


class ToRGB(nn.Module):
    """
    ToRGB layer for the StyleGAN2 generator.
    Maps the feature map to RGB space.
    """

    def __init__(self, w_dim, in_channels):
        """
        Maps the feature map to RGB space.
        Args:
            w_dim (int): Dimension of the style vector w.
            in_channels (int): Number of input feature channels.
        """

        super().__init__()
        self.conv = Conv2dWeightModulate(in_channels, 3, kernel_size=1, demodulate=False) # 1x1 conv to RGB [batch_size, 3, H, W]
        self.to_style = EqualizedLinear(w_dim, in_channels) # Style modulation layer [batch_size, in_channels]
        self.bias = nn.Parameter(torch.zeros(3)) # RGB bias 

    def forward(self, x, w):
        """
        Maps the feature map to RGB space.
        Args:
            x (torch.Tensor): Input feature map of shape [batch_size, in_channels, H, W].
            w (torch.Tensor): Style vector of shape [batch_size, w_dim].
        Returns:
            torch.Tensor: RGB output of shape [batch_size, 3, H, W].
        """

        style = self.to_style(w) # Map w to style vector 
        x = self.conv(x, style) # Apply 1x1 conv to RGB
        return x + self.bias[None, :, None, None]  # Linear output before tanh [batch_size, 3, H, W]


class Generator(nn.Module):
    """
    StyleGAN2 Generator
    Encapsulates the full generator architecture with progressive blocks.
    
    """
    def __init__(self, log_res, w_dim, base_channels=64, max_channels=512):
        super().__init__()
        self.log_res = log_res
        channels = [min(max_channels, base_channels * (2 ** i)) for i in range(log_res - 2, -1, -1)]
        self.initial_const = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.initial_block = StyleBlock(w_dim, channels[0], channels[0])
        self.to_rgb0 = ToRGB(w_dim, channels[0])

        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList([self.to_rgb0])
        for i in range(1, len(channels)):
            self.blocks.append(
                nn.ModuleList([
                    StyleBlock(w_dim, channels[i-1], channels[i]),
                    StyleBlock(w_dim, channels[i], channels[i])
                ])
            )
            self.to_rgbs.append(ToRGB(w_dim, channels[i]))

    def forward(self, ws, noises):
        batch = ws.shape[0]
        x = self.initial_const.expand(batch, -1, -1, -1)
        x = self.initial_block(x, ws[:, 0], noises[0][0])
        rgb = self.to_rgb0(x, ws[:, 0])

        for i, (block1, block2) in enumerate(self.blocks):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = block1(x, ws[:, 2*i+1], noises[i+1][0])
            x = block2(x, ws[:, 2*i+2], noises[i+1][1])
            rgb = F.interpolate(rgb, scale_factor=2, mode='nearest') + self.to_rgbs[i+1](x, ws[:, 2*i+2])

        return torch.tanh(rgb)


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, demodulate=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_channels, in_channels, kernel_size, kernel_size])

    def forward(self, x, style):
        batch, in_c, h, w = x.shape
        style = style[:, None, :, None, None]  # [B,1,C,1,1]
        weight = self.weight()[None] * style   # [B, out, in, k, k]

        if self.demodulate:
            demod = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4)) + self.eps)
            weight = weight * demod[:, :, None, None, None]

        x = x.reshape(1, -1, h, w)
        weight = weight.reshape(batch * weight.shape[1], weight.shape[2], weight.shape[3], weight.shape[4])
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        out = out.reshape(batch, -1, h, w)
        return out


class DiscriminatorBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Residual path
        self.residual = nn.Sequential(
            nn.AvgPool2d(2),
            EqualizedConv2d(in_channels, out_channels, kernel_size=1)
        )

        # Main path
        self.conv1 = EqualizedConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = nn.AvgPool2d(2)
        self.scale = 1 / sqrt(2)  # Rescale after residual sum

    def forward(self, x):
        residual = self.residual(x)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.downsample(x)
        return (x + residual) * self.scale


class Discriminator(nn.Module):
    def __init__(self, log_res, base_channels=64, max_channels=512):
        super().__init__()
        channels = [min(max_channels, base_channels * (2 ** i)) for i in range(log_res - 1)]

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, channels[0], kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(DiscriminatorBlock(channels[i], channels[i + 1]))
        self.blocks = nn.Sequential(*blocks)

        # Minibatch standard deviation
        self.mbstd = True

        # Final layers
        final_channels = channels[-1]
        self.final_conv = EqualizedConv2d(final_channels + 1 if self.mbstd else final_channels, final_channels, kernel_size=3, padding=1)
        self.final_linear = EqualizedLinear(4 * 4 * final_channels, 1)  # assumes 4x4 final resolution

    def minibatch_stddev(self, x, group_size=4):
        """
        Add a single channel containing the stddev of each feature over the minibatch (per group)
        """
        batch, c, h, w = x.shape
        group_size = min(group_size, batch)
        y = x.view(group_size, -1, c, h, w)  # [G, B//G, C, H, W]
        y = y - y.mean(dim=0, keepdim=True)
        y = (y ** 2).mean(dim=0)
        y = torch.sqrt(y + 1e-8)
        y = y.mean(dim=[1,2,3], keepdim=True)  # average over channels + spatial dims
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)
        if self.mbstd:
            x = self.minibatch_stddev(x)
        x = self.final_conv(x)
        x = torch.flatten(x, 1)
        return self.final_linear(x)


class ConditionalDiscriminator(nn.Module):
    """
    Discriminator with projection-based conditioning.
    Uses the method from "cGANs with Projection Discriminator" paper.
    """
    def __init__(self, log_res, num_classes=2, base_channels=64, max_channels=512):
        super().__init__()
        channels = [min(max_channels, base_channels * (2 ** i)) for i in range(log_res - 1)]
        
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, channels[0], kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(DiscriminatorBlock(channels[i], channels[i + 1]))
        self.blocks = nn.Sequential(*blocks)
        
        self.mbstd = True
        final_channels = channels[-1]
        
        self.final_conv = EqualizedConv2d(
            final_channels + 1 if self.mbstd else final_channels,
            final_channels,
            kernel_size=3,
            padding=1
        )
        
        # Unconditional output
        self.final_linear = EqualizedLinear(4 * 4 * final_channels, 1)
        
        # Class projection for conditioning
        self.class_embed = nn.Embedding(num_classes, 4 * 4 * final_channels)
    
    def minibatch_stddev(self, x, group_size=4):
        batch, c, h, w = x.shape
        group_size = min(group_size, batch)
        y = x.view(group_size, -1, c, h, w)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y ** 2).mean(dim=0)
        y = torch.sqrt(y + 1e-8)
        y = y.mean(dim=[1,2,3], keepdim=True)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)
    
    def forward(self, x, class_labels):
        """
        Args:
            x: [batch_size, 3, H, W]
            class_labels: [batch_size] with values 0 (AD) or 1 (NC)
        """
        x = self.from_rgb(x)
        x = self.blocks(x)
        
        if self.mbstd:
            x = self.minibatch_stddev(x)
        
        x = self.final_conv(x)
        features = torch.flatten(x, 1)
        
        # Unconditional score
        out = self.final_linear(features)
        
        # Add projection conditioning
        class_embed = self.class_embed(class_labels)
        projection = (features * class_embed).sum(dim=1, keepdim=True)
        
        return out + projection


class EqualizedWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.scale = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.scale

class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=0.0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.zeros(out_features) + bias)

    def forward(self, x):
        return F.linear(x, self.weight(), bias=self.bias)

class EqualizedConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding=0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class PathLengthPenalty(nn.Module):
    def __init__(self, decay=0.99):
        super().__init__()
        self.decay = decay
        self.register_buffer('pl_mean', torch.zeros(1))

    def forward(self, fake_images, w):
        noise = torch.randn_like(fake_images) / sqrt(fake_images.shape[2] * fake_images.shape[3])
        output = (fake_images * noise).sum()
        gradients = torch.autograd.grad(outputs=output, inputs=w, create_graph=True)[0]
        pl_lengths = torch.sqrt((gradients ** 2).sum(dim=-1).mean(dim=1))

        # Exponential moving average of path length
        self.pl_mean = self.decay * self.pl_mean + (1 - self.decay) * pl_lengths.mean().detach()
        loss = ((pl_lengths - self.pl_mean) ** 2).mean()
        return loss
    
def initialize_weights(module):
    """
    Apply improved weight initialization to the entire model.
    Call this after creating Generator and Discriminator.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # He initialization for conv layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # He initialization for linear layers
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, EqualizedWeight):
            # Improved initialization for EqualizedWeight
            with torch.no_grad():
                m.weight.normal_(0, 1.0)

# Add this function to improve ToRGB initialization specifically
def initialize_toRGB_bias(gen, target_mean=-0.5):
    """
    Initialize ToRGB biases to shift output toward better range.
    This helps early training produce more visible images.
    """
    with torch.no_grad():
        # Access ToRGB layers
        if hasattr(gen, 'to_rgb0'):
            gen.to_rgb0.bias.fill_(target_mean)
        
        if hasattr(gen, 'to_rgbs'):
            for to_rgb in gen.to_rgbs:
                to_rgb.bias.fill_(target_mean)