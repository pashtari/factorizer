import torch

import factorizer as ft


# %% Swin Metricize
size = (40, 24, 24, 24)
patch_size = (6, 6, 6)
head_dim = 8
x = torch.rand(2, *size)

swin_metricize = ft.SwinMatricize(
    (None, *size), head_dim=head_dim, patch_size=patch_size
)
y = swin_metricize.forward(x)
z = swin_metricize.inverse_forward(y)

print(torch.equal(x, z))

