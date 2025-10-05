""" 
modules.py
Conditional StyleGAN2 Model Implementation for ADNI Dataset
- Model architecture adapted for conditional generation (AD vs NC)
- Includes conditional mapping network and projection-based discriminator
- Implementation based on the original StyleGAN2 paper and PyTorch framework
Author: Tyreece Paul
"""

import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import numpy as np


class ConditionalMappingNetwork(nn.Module):
    """
    Conditional mapping network for class-conditional StyleGAN/StyleGAN2.

    This network extends the standard mapping network by incorporating class
    information (e.g., AD vs NC labels). It maps the initial latent vector z
    along with a class label to an intermediate latent vector w that is
    conditioned on the provided class. This allows the generator to produce
    class-specific outputs while maintaining disentanglement in the latent space.
    """

    def __init__(self, z_dim: int, w_dim: int, num_classes: int = 2, num_layers: int = 8):
        """
        Initializes the ConditionalMappingNetwork.

        The network uses learned class embeddings to incorporate categorical
        information into the latent space. The concatenated representation
        (z + class embedding) is then passed through a stack of fully-connected
        layers with LeakyReLU activations and equalized learning rates.

        Args:
            z_dim (int): Dimension of the input latent vector z.
            w_dim (int): Dimension of the output intermediate latent vector w.
            num_classes (int, optional): Number of class categories (e.g., 2 for
                binary classification like AD vs NC). Defaults to 2.
            num_layers (int, optional): The number of fully-connected layers
                in the mapping network. Defaults to 8.
        """

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
        Forward pass through the conditional mapping network.
        The process involves:
        1. Normalizing the input z to unit length.
        2. Converting discrete class labels into continuous learned embeddings.
        3. Combining the normalized z with the class embedding to form a conditioned latent representation.
        4. Passing the concatenated vector through the stack of linear layers and non-linearities.

        Args:
            z (torch.Tensor): Input latent vector z of shape [batch_size, z_dim].
            class_labels (torch.Tensor): Class labels of shape [batch_size] with
            integer values (e.g., 0 for AD, 1 for NC).

        Returns:
            torch.Tensor: Output intermediate latent vector w of shape
                [batch_size, w_dim] conditioned on the provided class labels.
        """
        
        # Normalize z
        z = z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)
        
        # Get class embedding and concatenate
        class_embed = self.class_embed(class_labels)
        z_cond = torch.cat([z, class_embed], dim=1)
        
        return self.mapping(z_cond)


class StyleBlock(nn.Module):
    """
    Core building block of the StyleGAN2 generator.

    This block applies style modulation to a convolutional layer, injects learned
    noise for stochastic variation, and uses an activation function. It is the
    primary mechanism for controlling the visual features of the generated image
    at a specific resolution.
    """
    def __init__(self, w_dim: int, in_channels: int, out_channels: int):
        """
        Initializes the StyleBlock.

        Args:
            w_dim (int): Dimension of the style vector w.
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
        """
        super().__init__()
        # Modulated convolution layer that applies the style
        self.conv = Conv2dWeightModulate(in_channels, out_channels, kernel_size=3)
        # Linear layer to project w to the style space of the convolution
        self.to_style = EqualizedLinear(w_dim, in_channels)
        # Per-channel bias term, applied after convolution
        self.bias = nn.Parameter(torch.zeros(out_channels))
        # Scaler for the injected noise, learned per-channel
        self.noise_scale = nn.Parameter(torch.zeros(1))
        # LeakyReLU activation
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the style block.

        Args:
            x (torch.Tensor): Input feature map of shape [batch_size, in_channels, H, W].
            w (torch.Tensor): Style vector of shape [batch_size, w_dim].
            noise (torch.Tensor, optional): Noise tensor of shape [batch_size, 1, H, W]
                to inject for stochastic details. Defaults to None.

        Returns:
            torch.Tensor: Output feature map of shape [batch_size, out_channels, H, W].
        """
        # Project w to get the style modulation vector
        style = self.to_style(w)
        # Apply style-modulated convolution
        x = self.conv(x, style)
        # Inject learned noise for stochasticity
        if noise is not None:
            x = x + self.noise_scale * noise
        # Add bias and apply activation
        x = x + self.bias[None, :, None, None]
        return self.activation(x)


class ToRGB(nn.Module):
    """
    Converts feature maps from a generator block into an RGB image.

    This layer is used at each resolution level of the generator to produce a
    corresponding RGB output. These outputs are upsampled and added together
    to form the final image, allowing features from all resolutions to contribute.
    """

    def __init__(self, w_dim: int, in_channels: int):
        """
        Initializes the ToRGB layer.

        Args:
            w_dim (int): Dimension of the style vector w.
            in_channels (int): Number of input feature channels from the generator block.
        """
        super().__init__()
        # 1x1 modulated convolution to map features to 3 RGB channels
        self.conv = Conv2dWeightModulate(in_channels, 3, kernel_size=1, demodulate=False)
        # Linear layer to project w to the style space for the 1x1 convolution
        self.to_style = EqualizedLinear(w_dim, in_channels)
        # Per-channel bias for the final RGB output
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to generate an RGB image from feature maps.

        Args:
            x (torch.Tensor): Input feature map of shape [batch_size, in_channels, H, W].
            w (torch.Tensor): Style vector of shape [batch_size, w_dim].

        Returns:
            torch.Tensor: Raw RGB output of shape [batch_size, 3, H, W]. This is a
                linear output before the final tanh activation.
        """
        # Project w to get the style modulation vector
        style = self.to_style(w)
        # Apply 1x1 modulated convolution to get RGB channels
        x = self.conv(x, style)
        # Add bias
        return x + self.bias[None, :, None, None]


class Generator(nn.Module):
    """
    StyleGAN2 Generator Architecture.

    This class assembles the full generator, starting from a learned constant
    input and progressively generating higher-resolution feature maps using a
    series of StyleBlocks. At each resolution, a ToRGB layer produces an image,
    and these are combined to form the final output.
    """
    def __init__(self, log_res: int, w_dim: int, base_channels: int = 64, max_channels: int = 512):
        """
        Initializes the Generator.

        Args:
            log_res (int): The base-2 logarithm of the output image resolution (e.g., 8 for 256x256).
            w_dim (int): Dimension of the intermediate latent vector w.
            base_channels (int, optional): The number of channels at the highest resolution.
                Channels increase for lower resolutions. Defaults to 64.
            max_channels (int, optional): The maximum number of channels in any layer. Defaults to 512.
        """
        super().__init__()
        self.log_res = log_res
        # Define channel counts for each resolution level, from high-res to low-res
        channels = [min(max_channels, base_channels * (2 ** i)) for i in range(log_res - 2, -1, -1)]

        # Initial 4x4 block
        self.initial_const = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.initial_block = StyleBlock(w_dim, channels[0], channels[0])
        self.to_rgb0 = ToRGB(w_dim, channels[0])

        # Subsequent blocks for 8x8 up to final resolution
        self.blocks = nn.ModuleList()
        self.to_rgbs = nn.ModuleList([self.to_rgb0])
        for i in range(1, len(channels)):
            # Each block upsamples and applies two StyleBlocks
            self.blocks.append(
                nn.ModuleList([
                    StyleBlock(w_dim, channels[i-1], channels[i]),
                    StyleBlock(w_dim, channels[i], channels[i])
                ])
            )
            self.to_rgbs.append(ToRGB(w_dim, channels[i]))

    def forward(self, ws: torch.Tensor, noises: list) -> torch.Tensor:
        """
        Forward pass through the generator.

        Args:
            ws (torch.Tensor): A stack of style vectors w for each layer, of shape
                [batch_size, num_layers, w_dim].
            noises (list): A list of noise tensors, one for each resolution level.

        Returns:
            torch.Tensor: The final generated image of shape [batch_size, 3, H, W],
                with pixel values scaled by tanh to [-1, 1].
        """
        batch = ws.shape[0]
        # Start with the learned constant, expanded to batch size
        x = self.initial_const.expand(batch, -1, -1, -1)
        # Process through the initial 4x4 block
        x = self.initial_block(x, ws[:, 0], noises[0][0])
        rgb = self.to_rgb0(x, ws[:, 0])

        # Process through the progressive upsampling blocks
        for i, (block1, block2) in enumerate(self.blocks):
            # Upsample feature map
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            # Apply the two StyleBlocks for this resolution
            x = block1(x, ws[:, 2*i+1], noises[i+1][0])
            x = block2(x, ws[:, 2*i+2], noises[i+1][1])
            # Upsample the RGB output from the previous level and add the new RGB contribution
            rgb = F.interpolate(rgb, scale_factor=2, mode='nearest') + self.to_rgbs[i+1](x, ws[:, 2*i+2])

        # Apply tanh activation to scale the final output to [-1, 1]
        return torch.tanh(rgb)


class Conv2dWeightModulate(nn.Module):
    """
    Modulated 2D Convolution Layer from StyleGAN2.

    This layer implements the core mechanism of style-based generation. Instead
    of learning convolutional weights directly, it modulates a single learned
    weight tensor using a style vector derived from w. This allows for powerful
    and fine-grained control over the generated image features.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, demodulate: bool = True, eps: float = 1e-8):
        """
        Initializes the modulated convolution layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            demodulate (bool, optional): Whether to apply weight demodulation to
                preserve the output feature map's standard deviation. Defaults to True.
            eps (float, optional): A small epsilon for numerical stability in demodulation.
                Defaults to 1e-8.
        """
        super().__init__()
        self.eps = eps
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        # A single, learnable weight tensor shared across all samples in a batch
        self.weight = EqualizedWeight([out_channels, in_channels, kernel_size, kernel_size])

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight modulation.

        Args:
            x (torch.Tensor): Input feature map of shape [batch, in_channels, H, W].
            style (torch.Tensor): Style vector of shape [batch, in_channels], used to
                modulate the convolutional weights.

        Returns:
            torch.Tensor: Output feature map of shape [batch, out_channels, H, W].
        """
        batch, in_c, h, w = x.shape
        # Reshape style for broadcasting: [B, 1, C_in, 1, 1]
        style = style[:, None, :, None, None]
        # Modulate weights: [B, C_out, C_in, k, k]
        weight = self.weight()[None] * style

        # Demodulation (optional but standard in StyleGAN2)
        if self.demodulate:
            # Calculate per-filter standard deviation
            demod = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4)) + self.eps)
            # Normalize weights
            weight = weight * demod[:, :, None, None, None]

        # Reshape for grouped convolution
        x = x.reshape(1, -1, h, w)
        weight = weight.reshape(batch * weight.shape[1], weight.shape[2], weight.shape[3], weight.shape[4])
        # Perform convolution as a grouped convolution for efficiency
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        # Reshape back to standard batch format
        out = out.reshape(batch, -1, h, w)
        return out


class DiscriminatorBlock(nn.Module):
    """
    A residual block for the StyleGAN2 discriminator.

    This block processes feature maps, downsamples them, and uses a residual
    connection to stabilize training and improve gradient flow.
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the DiscriminatorBlock.

        Args:
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
        """
        super().__init__()
        # Residual path for downsampling the input
        self.residual = nn.Sequential(
            nn.AvgPool2d(2),
            EqualizedConv2d(in_channels, out_channels, kernel_size=1)
        )

        # Main convolutional path
        self.conv1 = EqualizedConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = nn.AvgPool2d(2)
        # Rescale factor to maintain variance after residual connection
        self.scale = 1 / sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator block.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map, downsampled by a factor of 2.
        """
        # Calculate residual connection
        residual = self.residual(x)
        # Process through main path
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.downsample(x)
        # Add residual and rescale
        return (x + residual) * self.scale


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator with Projection-Based Conditioning.

    This discriminator evaluates the "realness" of an image while also
    considering its class label (e.g., AD or NC). It uses the projection
    discriminator method, which adds a term based on the dot product of
    image features and class embeddings to the final score.
    """
    def __init__(self, log_res: int, num_classes: int = 2, base_channels: int = 64, max_channels: int = 512):
        """
        Initializes the ConditionalDiscriminator.

        Args:
            log_res (int): Base-2 logarithm of the input image resolution.
            num_classes (int, optional): Number of classes for conditioning. Defaults to 2.
            base_channels (int, optional): Number of channels at the highest resolution. Defaults to 64.
            max_channels (int, optional): Maximum number of channels in any layer. Defaults to 512.
        """
        super().__init__()
        # Define channel counts for each resolution level
        channels = [min(max_channels, base_channels * (2 ** i)) for i in range(log_res - 1)]
        
        # Initial layer to process RGB image
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, channels[0], kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Sequence of downsampling blocks
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(DiscriminatorBlock(channels[i], channels[i + 1]))
        self.blocks = nn.Sequential(*blocks)
        
        # Minibatch Standard Deviation for increased variation
        self.mbstd = True
        final_channels = channels[-1]
        
        # Final convolutional layer
        self.final_conv = EqualizedConv2d(
            final_channels + 1 if self.mbstd else final_channels,
            final_channels,
            kernel_size=3,
            padding=1
        )
        
        # Final linear layer for the unconditional real/fake score
        self.final_linear = EqualizedLinear(4 * 4 * final_channels, 1)
        
        # Embedding layer for class projection
        self.class_embed = nn.Embedding(num_classes, 4 * 4 * final_channels)
    
    def minibatch_stddev(self, x: torch.Tensor, group_size: int = 4) -> torch.Tensor:
        """
        Calculates and appends the minibatch standard deviation as a new feature channel.

        This encourages the generator to produce more diverse outputs by penalizing
        the discriminator if a batch of generated images is too uniform.

        Args:
            x (torch.Tensor): Input feature map of shape [batch, C, H, W].
            group_size (int, optional): Size of groups to calculate stddev over. Defaults to 4.

        Returns:
            torch.Tensor: Feature map with the appended stddev channel, shape [batch, C+1, H, W].
        """
        batch, c, h, w = x.shape
        group_size = min(group_size, batch)
        # Split batch into groups
        y = x.view(group_size, -1, c, h, w)
        # Calculate stddev over the group
        y = y - y.mean(dim=0, keepdim=True)
        y = (y ** 2).mean(dim=0)
        y = torch.sqrt(y + 1e-8)
        # Average stddev over channels and spatial dimensions
        y = y.mean(dim=[1,2,3], keepdim=True)
        # Repeat to match spatial dimensions and concatenate
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)
    
    def forward(self, x: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the conditional discriminator.

        Args:
            x (torch.Tensor): Input image of shape [batch_size, 3, H, W].
            class_labels (torch.Tensor): Class labels of shape [batch_size].

        Returns:
            torch.Tensor: A single score per image, where a higher score indicates
                a higher probability of being real and belonging to the correct class.
        """
        # Process image through convolutional layers
        x = self.from_rgb(x)
        x = self.blocks(x)
        
        # Apply minibatch standard deviation
        if self.mbstd:
            x = self.minibatch_stddev(x)
        
        # Final convolution and feature flattening
        x = self.final_conv(x)
        features = torch.flatten(x, 1)
        
        # Get the unconditional real/fake score
        out = self.final_linear(features)
        
        # Add the class-conditional projection term
        class_embed = self.class_embed(class_labels)
        projection = (features * class_embed).sum(dim=1, keepdim=True)
        
        return out + projection


class EqualizedWeight(nn.Module):
    """
    A learnable weight parameter with equalized learning rate scaling.

    This technique, introduced in StyleGAN, helps to stabilize training by
    scaling weights at runtime, making the model less sensitive to the
    initialization scale.
    """
    def __init__(self, shape: list):
        """
        Initializes the EqualizedWeight.

        Args:
            shape (list): The shape of the weight tensor.
        """
        super().__init__()
        # He initializer scale, calculated from the fan-in
        self.scale = 1 / sqrt(np.prod(shape[1:]))
        # The learnable weight parameter, initialized from a standard normal distribution
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self) -> torch.Tensor:
        """
        Returns the scaled weight tensor.
        """
        return self.weight * self.scale


class EqualizedLinear(nn.Module):
    """
    A fully-connected layer with an equalized learning rate.
    """
    def __init__(self, in_features: int, out_features: int, bias: float = 0.0):
        """
        Initializes the EqualizedLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (float, optional): Initial value for the bias term. Defaults to 0.0.
        """
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.zeros(out_features) + bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation.
        """
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    """
    A 2D convolutional layer with an equalized learning rate.
    """
    def __init__(self, in_features: int, out_features: int, kernel_size: int, padding: int = 0):
        """
        Initializes the EqualizedConv2d layer.

        Args:
            in_features (int): Number of input channels.
            out_features (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int, optional): Padding to add to the input. Defaults to 0.
        """
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the 2D convolution.
        """
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class PathLengthPenalty(nn.Module):
    """
    Implements the Path Length Regularization for StyleGAN2.

    This regularization technique encourages a smoother and more disentangled
    latent space (w) by penalizing large changes in the generated image when
    making small steps in the latent space.
    """
    def __init__(self, decay: float = 0.99):
        """
        Initializes the PathLengthPenalty module.

        Args:
            decay (float, optional): The decay rate for the exponential moving
                average of path lengths. Defaults to 0.99.
        """
        super().__init__()
        self.decay = decay
        # Buffer to store the exponential moving average of path lengths
        self.register_buffer('pl_mean', torch.zeros(1))

    def forward(self, fake_images: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Calculates the path length penalty loss.

        Args:
            fake_images (torch.Tensor): A batch of generated images.
            w (torch.Tensor): The corresponding latent vectors used to generate the images.

        Returns:
            torch.Tensor: The path length penalty loss.
        """
        # Project random noise onto the image to get a scalar output
        noise = torch.randn_like(fake_images) / sqrt(fake_images.shape[2] * fake_images.shape[3])
        output = (fake_images * noise).sum()
        # Calculate the gradient of the output with respect to w
        gradients = torch.autograd.grad(outputs=output, inputs=w, create_graph=True)[0]
        # Calculate the L2 norm of the gradients (per-sample path lengths)
        pl_lengths = torch.sqrt((gradients ** 2).sum(dim=-1).mean(dim=1))

        # Update the exponential moving average of path lengths
        self.pl_mean = self.decay * self.pl_mean + (1 - self.decay) * pl_lengths.mean().detach()
        # The loss is the mean squared difference between individual path lengths and the moving average
        loss = ((pl_lengths - self.pl_mean) ** 2).mean()
        return loss
