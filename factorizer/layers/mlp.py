from torch import nn
from torch.nn.modules.utils import _pair

from .linear import Linear


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        ratio=2,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = (
            int(ratio * in_channels) if hidden_channels is None else hidden_channels
        )
        dropout = _pair(dropout)

        self.block = nn.Sequential(
            Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout[0]),
            Linear(hidden_channels, out_channels),
            nn.Dropout(dropout[1]),
        )

    def forward(self, x):
        return self.block(x)
