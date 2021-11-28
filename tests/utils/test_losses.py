import torch

from factorizer.utils.losses import dice_loss, dicece_loss, dicefocal_loss

y = torch.randint(0, 2, dtype=torch.float32, size=(1, 1, 128, 128, 128))
p = (
    torch.randn((1, 1, 128, 128, 128)),
    torch.randn((1, 1, 64, 64, 64)),
    torch.randn((1, 1, 32, 32, 32)),
)

loss = dice_loss(p, y)
loss = dicece_loss(p, y)
loss = dicefocal_loss(p, y)
