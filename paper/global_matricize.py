import torch
from torch import nn
from einops.layers.torch import Rearrange


class GlobalMatricize(nn.Module):
    def __init__(self, size, head_dim=8):
        super().__init__()
        C, H, W, D = size
        dims = dict(c2=head_dim, h=H, w=W, d=D)

        eq = "b (c1 c2) h w d -> (b c1) c2 (h w d)"
        self.unfold = Rearrange(eq, **dims)

        eq = "(b c1) c2 (h w d) -> b (c1 c2) h w d"
        self.fold = Rearrange(eq, **dims)

    def forward(self, x):
        # x: (B, C, H, W, D)
        return self.unfold(x)

    def inverse_forward(self, x):
        # x: (B, C, L)
        return self.fold(x)
