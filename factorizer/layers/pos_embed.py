import math

import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, channels, spatial_size, **kwargs):
        super().__init__()
        spatial_dims = len(spatial_size)
        freqs = torch.exp(torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))
        theta = 0
        for dim, size in enumerate(spatial_size):
            p_size = [size if j == dim else 1 for j in range(spatial_dims)]
            x = torch.arange(size).reshape(1, 1, *p_size)
            omega = freqs.reshape(1, -1, *(spatial_dims * [1]))
            theta += omega * x

        cos_pe = torch.cos(theta)
        sin_pe = torch.sin(theta)
        pe = torch.cat((cos_pe, sin_pe), dim=1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = x + self.pe
        return out


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(self, channels, spatial_size, **kwargs):
        super().__init__()
        spatial_dims = len(spatial_size)
        freqs = torch.exp(torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))
        theta = 0.0
        for dim, size in enumerate(spatial_size):
            p_size = [size if j == dim else 1 for j in range(spatial_dims)]
            x = torch.arange(size).reshape(1, 1, *p_size)
            omega = freqs.reshape(1, -1, *(spatial_dims * [1]))
            theta = theta + omega * x

        theta = torch.cat((theta, theta), dim=1)

        cos = torch.cos(theta)
        sin = torch.sin(theta)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        d = x.shape[1]
        x_1, x_2 = x[:, : d // 2, ...], x[:, d // 2 :, ...]
        x_half = torch.cat((-x_2, x_1), dim=1)
        out = self.cos * x + self.sin * x_half
        return out


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding."""

    def __init__(self, channels, spatial_size, **kwargs):
        super().__init__()
        self.pos = nn.Parameter(torch.empty(1, channels, *spatial_size))
        nn.init.normal_(self.pos, std=1.0)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = x + self.pos
        return out


# an alias for compatibality with the previous version
PosEmbed = PositionalEmbedding


class AxialPositionalEmbedding(nn.Module):
    """Learnable axial positional embedding."""

    def __init__(self, channels, spatial_size, **kwargs):
        super().__init__()
        self.channels = channels
        self.pe = nn.ParameterList([])
        for dim, size in enumerate(spatial_size):
            p_size = [size if j == dim else 1 for j in range(len(spatial_size))]
            p = nn.Parameter(torch.empty(1, channels, *p_size))
            nn.init.normal_(p, std=1.0)
            self.pe.append(p)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = x
        for p in self.pe:
            out = out + p

        return out
