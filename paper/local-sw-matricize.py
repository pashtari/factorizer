import torch
from torch import nn
from einops.layers.torch import Rearrange


class LocalMatricize(nn.Module):
    def __init__(self, size, head_dim=8, patch_size=(8, 8, 8)):
        super().__init__()
        C, H, W, D = size
        c2, (h2, w2, d2) = head_dim, patch_size
        c1, h1, w1, d1 = C // c2, H // h2, W // w2, D // d2
        dims = dict(c1=c1, h1=h1, w1=w1, d1=d1, c2=c2, h2=h2, w2=w2, d2=d2)

        eq = "b (c1 c2) (h1 h2) (w1 w2) (d1 d2) -> (b c1 h1 w1 d1) c2 (h2 w2 d2)"
        self.unfold = Rearrange(eq, **dims)

        eq = "(b c1 h1 w1 d1) c2 (h2 w2 d2) -> b (c1 c2) (h1 h2) (w1 w2) (d1 d2)"
        self.fold = Rearrange(eq, **dims)

    def forward(self, x):
        # x: (B, C, H, W, D)
        return self.unfold(x)

    def inverse_forward(self, x):
        # x: (B, C, L)
        return self.fold(x)


class SwinMatricize(nn.Module):
    def __init__(
        self, size, head_dim=8, patch_size=(8, 8, 8), shifts=(4, 4, 4)
    ):
        super().__init__()
        self.local_matricize = LocalMatricize(size, head_dim, patch_size)
        self.shifts = shifts
        self.shifts_inv = tuple(-s for s in patch_size)

    def forward(self, x):
        # x: (B, C, H, W, D)
        out1 = self.local_matricize(x)
        out2 = torch.roll(x, self.shifts, (2, 3, 4))
        out2 = self.local_matricize(out2)
        return torch.cat((out1, out2))

    def inverse_forward(self, x):
        # x: (B, C, L)
        B = x.shape[0]
        out1 = self.local_matricize.inverse_forward(x[: (B // 2)])
        out2 = self.local_matricize.inverse_forward(x[(B // 2) :])
        out2 = torch.roll(out2, self.shifts_inv, (2, 3, 4))
        return 0.5 * (out1 + out2)
