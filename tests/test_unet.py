# %%
import torch
from factorizer import (
    StemBlock,
    Same,
    HeadBlock,
    BasicBlock,
    PreActivationBlock,
    UNetEncoder,
    UNetDecoder,
    UNet,
    DualUNet,
)

# %% U-Net Encoder
image_size = (64, 64, 64)
x = torch.rand(1, 2, *image_size)
print(f"input shape: {x.shape}")

encoder = UNetEncoder(
    2, depth=(1, 2, 2, 2), width=(32, 64, 128, 256), block=Same(BasicBlock)
)
f = encoder(x)
out = f[-1].sum()
out.backward()

for i, out in enumerate(f):
    print(f"U-Net encoder: output_{i} shape: {tuple(out.shape)}")


# %% U-Net Decoder
decoder = UNetDecoder(
    256,
    depth=(1, 1, 1, 1),
    width=(128, 64, 32),
    strides=(2, 2, 2),
    upsample="tconv",
    block=Same(BasicBlock),
)
y = decoder(f)
for i, out in enumerate(y):
    print(f"U-Net decoder: output_{i} shape: {tuple(out.shape)}")

# %% U-Net
image_size = (64, 64, 64)
x = torch.rand(1, 2, *image_size)
print(f"input shape: {x.shape}")

unet = UNet(
    2,
    3,
    stem_width=None,
    encoder_depth=(1, 1, 1, 1),
    encoder_width=(32, 64, 128, 256),
    strides=(1, 2, 2, 2),
    decoder_depth=(1, 1, 1),
    upsample="tconv",
    stem=StemBlock,
    stem_params=None,
    block=Same(PreActivationBlock),
    head=HeadBlock,
    head_params=None,
    num_deep_supr=3,
)

unet.zero_grad()
y = unet(x)
out = y[0].sum()
out.backward()

for i, out in enumerate(y):
    print(f"U-Net: output_{i} shape: {tuple(out.shape)}")


# %% Dual U-Net
image_size = (64, 64, 64)
x = torch.rand(1, 2, *image_size)
print(f"input shape: {x.shape}")

dual_unet = DualUNet(
    out_channels=3,
    stem_width=None,
    encoder_depth=(1, 1, 1, 1),
    encoder_width=(32, 64, 128, 256),
    strides=(1, 2, 2, 2),
    decoder_depth=(1, 1, 1),
    upsample="tconv",
    stem=StemBlock,
    stem_params=None,
    block=Same(PreActivationBlock),
    head=HeadBlock,
    head_params=None,
    num_deep_supr=3,
)

dual_unet.zero_grad()
y = dual_unet(x)
out = y[0].sum()
out.backward()

for i, out in enumerate(y):
    print(f"Dual U-Net: output_{i} shape: {tuple(out.shape)}")
