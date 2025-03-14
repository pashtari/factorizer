from typing import Sequence
import math

import torch
from torch import nn

from .utils.helpers import as_tuple, partialize
from .layers.conv import DoubleConv


class Same(object):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def __getitem__(self, *args, **kwargs):
        return self.block


class UNetStage(nn.Module):
    """U-Net block for one stage."""

    def __init__(self, in_channels, out_channels, depth=1, block=DoubleConv, **kwargs):
        super().__init__()
        block = partialize(block)
        self.blocks = nn.Sequential(block(in_channels, out_channels, **kwargs))

        for j in range(1, depth):
            self.blocks.append(block(out_channels, out_channels, **kwargs))

    def forward(self, x):
        out = self.blocks(x)
        return out


class UNetEncoderBlock(nn.Module):
    """U-Net encoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        stride=2,
        downsample=(nn.Conv3d, {"kernel_size": 2}),
        block=UNetStage,
        **kwargs,
    ):
        super().__init__()
        block = partialize(block)
        downsample = nn.Identity if math.prod(as_tuple(stride)) == 1 else downsample
        downsample = partialize(downsample)
        self.downsample = downsample(in_channels, out_channels, stride=2)
        self.block = block(out_channels, out_channels, depth=depth, **kwargs)

    def forward(self, x):
        out = self.downsample(x)
        out = self.block(out)
        return out


class UNetEncoder(nn.Module):
    """U-Net encoder."""

    def __init__(
        self,
        in_channels,
        out_channels=(32, 64, 128, 256, 512),
        depth=(1, 1, 1, 1, 1),
        strides=(1, 2, 2, 2, 2),
        downsample=None,
        block=Same(DoubleConv),
        **kwargs,
    ):
        super().__init__()
        channels = [in_channels, *out_channels]
        self.in_spatial_size = kwargs.get("spatial_size")
        self.blocks = nn.ModuleList()
        for i in range(len(out_channels)):
            if isinstance(kwargs.get("spatial_size"), Sequence):
                kwargs["spatial_size"] = tuple(
                    d // strides[i] for d in kwargs["spatial_size"]
                )

            self.blocks.append(
                UNetEncoderBlock(
                    channels[i],
                    channels[i + 1],
                    depth[i],
                    strides[i],
                    downsample,
                    block[i],
                    **kwargs,
                )
            )

        self.out_spatial_size = kwargs.get("spatial_size")

    def forward(self, x):
        out = [self.blocks[0](x)]
        for blk in self.blocks[1:]:
            out.append(blk(out[-1]))

        return out


class UNetDecoderBlock(nn.Module):
    """U-Net decoder block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        stride=2,
        upsample=(nn.ConvTranspose3d, {"kernel_size": 2}),
        block=UNetStage,
        **kwargs,
    ):
        super().__init__()
        upsample = partialize(upsample)
        block = partialize(block)
        self.upsample = upsample(in_channels, out_channels, stride=stride)
        self.block = block(2 * out_channels, out_channels, depth=depth, **kwargs)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        out = torch.cat([x2, x1], dim=1)
        out = self.block(out)
        return out


class UNetDecoder(nn.Module):
    """U-Net decoder."""

    def __init__(
        self,
        in_channels=(512, 256, 128, 64, 32),
        depth=(1, 1, 1, 1),
        strides=(2, 2, 2, 2),
        upsample=None,
        block=Same(DoubleConv),
        **kwargs,
    ):
        super().__init__()
        self.in_spatial_size = kwargs.get("spatial_size")
        self.blocks = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            if isinstance(kwargs.get("spatial_size"), Sequence):
                kwargs["spatial_size"] = tuple(
                    d * strides[i] for d in kwargs["spatial_size"]
                )
            self.blocks.append(
                UNetDecoderBlock(
                    in_channels[i],
                    in_channels[i + 1],
                    depth[i],
                    strides[i],
                    upsample,
                    block[i],
                    **kwargs,
                )
            )

        self.out_spatial_size = kwargs.get("spatial_size")

    def forward(self, x):
        out = x
        for i, blk in enumerate(self.blocks):
            i1 = -1 - i
            i2 = -2 - i
            out[i2] = blk(out[i1], out[i2])

        return out


class UNet(nn.Module):
    """Generic U-Net architecture."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_dims=3,
        spatial_size=None,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        block=None,
        upsample=None,
        head=None,
        num_deep_supr=False,
        **kwargs,
    ):
        super().__init__()

        self.spatial_dims = spatial_dims
        self.spatial_size = spatial_size

        conv = getattr(nn, f"Conv{self.spatial_dims}d")
        tconv = getattr(nn, f"ConvTranspose{self.spatial_dims}d")

        if stem in (None, nn.Identity):
            stem = nn.Identity
            stem_width = in_channels
        else:
            stem_width = encoder_width[0]

        if downsample is None:
            downsample = (conv, {"kernel_size": 2})

        if block is None:
            block = (
                DoubleConv,
                {"conv": (conv, {"kernel_size": 3, "padding": 1})},
            )
            block = Same(block)

        if upsample is None:
            upsample = (tconv, {"kernel_size": 2})

        if head is None:
            head = (conv, {"kernel_size": 1})

        stem = partialize(stem)
        head = partialize(head)

        self.stem = stem(in_channels, stem_width)
        self.encoder = UNetEncoder(
            stem_width,
            encoder_width,
            encoder_depth,
            strides,
            downsample,
            [block[i] for i in range(len(encoder_depth))],
            spatial_size=spatial_size,
            **kwargs,
        )
        self.decoder = UNetDecoder(
            encoder_width[::-1],
            decoder_depth,
            strides[::-1][: len(decoder_depth)],
            upsample,
            [block[i + len(encoder_depth)] for i in range(len(decoder_depth))],
            spatial_size=self.encoder.out_spatial_size,
            **kwargs,
        )
        if num_deep_supr in (False, None):
            self.num_deep_supr = False
            self.head = head(encoder_width[0], out_channels)
        else:
            self.num_deep_supr = 3 if num_deep_supr == True else num_deep_supr
            self.heads = nn.ModuleList()
            for j in range(self.num_deep_supr):
                self.heads.append(head(encoder_width[j], out_channels))

    def forward_features(self, x):
        out = self.stem(x)
        out = self.encoder(out)
        out = self.decoder(out)
        return out

    def forward(self, x):
        y = self.forward_features(x)

        if self.num_deep_supr:
            out = []
            for j, head in enumerate(self.heads):
                out.append(head(y[j]))
        else:
            out = self.head(y[0])

        return out
