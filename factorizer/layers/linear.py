from typing import Optional

import torch
from torch import nn


class Linear(nn.Module):
    """Linear layer for channels-first inputs.

    This layer applies a 1D convolution to the input after flattening
    all dimensions except the first two, effectively performing a linear
    transformation similar to `nn.Linear` but for channels-first inputs.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bias (bool, optional): If True, adds a learnable bias to the output.
            Default: True.
        device (Optional[torch.device], optional): The desired device of the
            weight and bias parameters. Default: None.
        dtype (Optional[torch.dtype], optional): The desired data type of the
            weight and bias parameters. Default: None.

    Shape:
        - Input: (N, C_{in}, *)
        - Output: (N, C_{out}, *)

    Example:
        >>> linear = Linear(in_channels=128, out_channels=64)
        >>> x = torch.randn(32, 128, 10, 10)
        >>> output = linear(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = self.flatten(x)  # Flatten spatial dimensions
        x = self.linear(x)
        x = x.view(original_shape[0], -1, *original_shape[2:])
        return x
