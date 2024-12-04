from typing import Optional

import torch
from torch import nn
from torch.nn.modules.utils import _pair

from .linear import Linear


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) for channels-first inputs.

    This module constructs feedforward neural network with one hidden layer, using a
    configurable ratio for hidden layer size, optional dropout between layers, and
    optional output size.

    Args:
        in_channels (int): Number of input channels.
        out_channels (Optional[int], optional): Number of output channels.
            Defaults to `in_channels` if not specified. Default: None.
        hidden_channels (Optional[int], optional): Number of hidden layer channels.
            Defaults to `in_channels * ratio` if not specified. Default: None.
        ratio (float, optional): Factor for determining hidden layer size when
            `hidden_channels` is not specified. Default: 2.0.
        dropout (float | tuple[float, float]], optional): Dropout rate(s) for the
            hidden and output layers. A single float applies the same rate to both layers.
            Default: 0.0.
        **kwargs: Additional keyword arguments passed to the `Linear` layer.

    Shape:
        - Input: (N, C_{in}, *)
        - Output: (N, C_{out}, *)

    Example:
        >>> mlp = MLP(in_channels=128, out_channels=64, ratio=3, dropout=(0.1, 0.2))
        >>> x = torch.randn(32, 128, 10, 10)
        >>> output = mlp(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        ratio: float = 3.0,
        dropout: float | tuple[float, float] = 0.0,
        **kwargs,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or int(ratio * in_channels)
        dropout = _pair(dropout)  # Ensure dropout is a tuple of two values

        self.block = nn.Sequential(
            Linear(in_channels, hidden_channels, **kwargs),
            nn.GELU(),
            nn.Dropout(dropout[0]),
            Linear(hidden_channels, out_channels, **kwargs),
            nn.Dropout(dropout[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
