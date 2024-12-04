import torch
from torch import nn


class LayerNorm(nn.Module):
    """Layer normalization for channels-first inputs.

    This module applies layer normalization to inputs with channels-first format
    (B, C, S1, S2, ..., Sp), where B is the batch size, C is the number of
    channels, and S1, S2, ..., Sp are spatial or temporal dimensions. The layer
    normalization is applied across the channel dimension.

    Args:
        dim (int): Number of channels.
        **kwargs: Additional keyword arguments passed to `nn.LayerNorm`.

    Example:
        >>> layer_norm = LayerNorm(3)
        >>> x = torch.randn(2, 3, 4, 4)  # Example input with shape (B, C, H, W)
        >>> output = layer_norm(x)
        >>> print(output.shape)
        torch.Size([2, 3, 4, 4])
    """

    def __init__(self, dim: int, **kwargs):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim, **kwargs)

    def forward(self, x):
        # x: (B, C, S1, S2, ..., Sp)
        out = torch.einsum("b c ... -> b ... c", x)
        out = self.norm(out)
        out = torch.einsum("b ... c -> b c ...", out)
        return out
