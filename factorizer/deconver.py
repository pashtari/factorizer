from torch import nn

from .utils import partialize
from .layers import Linear, LayerNorm, MLP
from .factorization import Deconv
from .unet import UNet


class DeconvMixer(nn.Module):
    """Blind deconvolutional mixing module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        act=nn.ReLU,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.in_proj = Linear(in_channels, out_channels, bias=False)
        self.deconv = Deconv(out_channels, **kwargs)
        self.act = partialize(act)()
        deconv_out_channels = self.deconv.groups * self.deconv.source_channels
        self.out_proj = Linear(deconv_out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, *)

        # input linear projection
        out = self.in_proj(x)

        # apply activation function
        out = self.act(out)

        # blind deconvolution
        out = self.deconv(out)

        # output linear projection
        out = self.out_proj(out)

        # dropout
        out = self.dropout(out)
        return out


class DeconverBlock(nn.Module):
    """Deconver Block."""

    def __init__(self, channels, norm=LayerNorm, dropout=0.0, mlp_ratio=4, **kwargs):
        super().__init__()

        self.norm1 = partialize(norm)(channels)
        self.dcm = DeconvMixer(channels, channels, **kwargs)

        self.norm2 = partialize(norm)(channels)
        self.mlp = MLP(channels, ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        out = x
        out = out + self.dcm(self.norm1(out))
        out = out + self.mlp(self.norm2(out))
        return out


# class DeconverBlock(nn.Module):
#     """Deconver Block."""

#     def __init__(
#         self, channels, norm=LayerNorm, act=nn.ReLU, mlp_ratio=4, dropout=0.0, **kwargs
#     ):
#         super().__init__()

#         self.norm = partialize(norm)(channels)
#         self.act = partialize(act)()
#         self.deconv = Deconv(channels, **kwargs)
#         deconv_out_channels = self.deconv.groups * self.deconv.source_channels
#         self.mlp = MLP(deconv_out_channels, channels, ratio=mlp_ratio, dropout=dropout)

#     def forward(self, x):
#         out = self.norm(x)
#         out = self.act(out)
#         out = self.deconv(out)
#         out = self.mlp(out)
#         out = x + out
#         return out


class DeconverStage(nn.Module):
    """Deconver block for one stage."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size=None,
        depth=1,
        adapter=(Linear, {"bias": False}),
        **kwargs,
    ):
        super().__init__()
        if in_channels != out_channels:
            self.adapter = partialize(adapter)(in_channels, out_channels)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                DeconverBlock(
                    out_channels,
                    **kwargs,
                )
            )

    def forward(self, x):
        # x: (B, C, *)
        out = self.adapter(x) if hasattr(self, "adapter") else x
        for blk in self.blocks:
            out = blk(out)

        return out


class Stem(nn.Sequential):
    def __init__(self, in_channels, out_channels, patch_size=(4, 4), norm=LayerNorm):
        spatial_dims = len(patch_size)
        _conv = getattr(nn, f"Conv{spatial_dims}d")
        _norm = partialize(norm)
        super().__init__(
            _conv(in_channels, out_channels, patch_size, stride=patch_size),
            _norm(out_channels),
        )


class Deconver(UNet):
    """Deconver for Segmentation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_dims=3,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        upsample=None,
        head=None,
        num_deep_supr=False,
        **kwargs,
    ):
        num_stages = len(encoder_depth) + len(decoder_depth)
        block = num_stages * [DeconverStage]
        if stem is None:
            stem = (
                getattr(nn, f"Conv{spatial_dims}d"),
                {"kernel_size": 3, "padding": 1, "bias": False},
            )
        super().__init__(
            in_channels,
            out_channels,
            spatial_dims=spatial_dims,
            encoder_depth=encoder_depth,
            encoder_width=encoder_width,
            strides=strides,
            decoder_depth=decoder_depth,
            stem=stem,
            downsample=downsample,
            block=block,
            upsample=upsample,
            head=head,
            num_deep_supr=num_deep_supr,
            **kwargs,
        )
