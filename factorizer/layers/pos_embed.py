from typing import Sequence

import math

import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for channels-first inputs.

    The embeddings are added to the input tensor to incorporate positional information.

    Args:
        channels (int): Number of channels in the positional embeddings.
        spatial_size (Sequence[int]): Spatial dimensions of the embedding.
    """

    def __init__(self, channels: int, spatial_size: Sequence[int]) -> None:
        super().__init__()
        spatial_dims = len(spatial_size)
        freqs = torch.exp(torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))
        theta = 0.0
        for dim, size in enumerate(spatial_size):
            p_size = [size if j == dim else 1 for j in range(spatial_dims)]
            x = torch.arange(size).reshape(1, 1, *p_size).float()
            omega = freqs.reshape(1, -1, *(spatial_dims * [1]))
            theta = theta + omega * x

        cos_pe = torch.cos(theta)
        sin_pe = torch.sin(theta)
        pe = torch.cat((cos_pe, sin_pe), dim=1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S1, S2, ..., Sp)
        return x + self.pe


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding for channels-first inputs.

    Args:
        channels (int): Number of channels in the positional embeddings.
        spatial_size (Sequence[int]): Spatial dimensions of the embedding.
    """

    def __init__(self, channels: int, spatial_size: Sequence[int]) -> None:
        super().__init__()
        spatial_dims = len(spatial_size)
        freqs = torch.exp(torch.arange(0, channels, 2) * (-math.log(10000.0) / channels))
        theta = 0.0
        for dim, size in enumerate(spatial_size):
            p_size = [size if j == dim else 1 for j in range(spatial_dims)]
            x = torch.arange(size).reshape(1, 1, *p_size).float()
            omega = freqs.reshape(1, -1, *(spatial_dims * [1]))
            theta = theta + omega * x

        theta = torch.cat((theta, theta), dim=1)
        self.register_buffer("cos", torch.cos(theta))
        self.register_buffer("sin", torch.sin(theta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S1, S2, ..., Sp)
        d = x.shape[1]
        x_1, x_2 = x[:, : d // 2, ...], x[:, d // 2 :, ...]
        x_half = torch.cat((-x_2, x_1), dim=1)
        return self.cos * x + self.sin * x_half


class PositionalEmbedding(nn.Module):
    """Learnable positional embedding for channels-first inputs.

    It generates learnable positional embeddings and adds them to the input tensor.

    Args:
        channels (int): Number of channels in the positional embeddings.
        spatial_size (Sequence[int]): Spatial dimensions of the embedding.
    """

    def __init__(self, channels: int, spatial_size: Sequence[int]) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.empty(1, channels, *spatial_size))
        nn.init.normal_(self.pos, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S1, S2, ..., Sp)
        return x + self.pos


# Alias for compatibility with previous versions
PosEmbed = PositionalEmbedding


class AxialPositionalEmbedding(nn.Module):
    """Learnable axial positional embedding for channels-first inputs.

    It generates learnable positional embeddings along each axis of the input tensor separately
    and adds them to the input tensor.

    Args:
        channels (int): Number of channels in the positional embeddings.
        spatial_size (Sequence[int]): Spatial dimensions of the embedding.
    """

    def __init__(self, channels: int, spatial_size: Sequence[int]) -> None:
        super().__init__()
        self.channels = channels
        self.pe = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        1,
                        channels,
                        *[size if j == dim else 1 for j in range(len(spatial_size))]
                    )
                )
                for dim, size in enumerate(spatial_size)
            ]
        )
        for p in self.pe:
            nn.init.normal_(p, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S1, S2, ..., Sp)
        out = x
        for p in self.pe:
            out = out + p
        return out
