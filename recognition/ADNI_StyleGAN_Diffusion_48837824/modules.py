import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

""" 
modules.py
StyleGAN2 Model Implementation
"""

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers=8):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(EqualizedLinear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        # PixelNorm
        x = z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)
        return self.mapping(x)


class GeneratorBlock(nn.Module):
    def __init__(self, W_DIM, in_features, out_features):
        super().__init__()

        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)

        self.to_rgb = ToRGB(W_DIM, out_features)

    def forward(self, x, w, noise):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb


class StyleBlock(nn.Module):
    def __init__(self, w_dim, in_channels, out_channels):
        super().__init__()
        self.conv = Conv2dWeightModulate(in_channels, out_channels, kernel_size=3)
        self.to_style = EqualizedLinear(w_dim, in_channels)  # Should match input channels for weight modulation
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.noise_scale = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w, noise=None):
        style = self.to_style(w)
        x = self.conv(x, style)
        if noise is not None:
            x = x + self.noise_scale * noise
        x = x + self.bias[None, :, None, None]
        return self.activation(x)


class ToRGB(nn.Module):
    def __init__(self, w_dim, in_channels):
        super().__init__()
        self.conv = Conv2dWeightModulate(in_channels, 3, kernel_size=1, demodulate=False)
        self.to_style = EqualizedLinear(w_dim, in_channels)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, x, w):
        style = self.to_style(w)
        x = self.conv(x, style)
        return x + self.bias[None, :, None, None]  # Linear output before tanh


class Generator(nn.Module):
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
