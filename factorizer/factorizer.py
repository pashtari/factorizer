from torch import nn

from .utils import partialize
from .layers import LayerNorm, Linear, MLP, PositionalEmbedding
from .factorization import Matricize, NMF
from .unet import UNet


class FactMixer(nn.Module):
    """Generic matrix/tensor factorization mixing module."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size,
        reshape=(Matricize, {"num_heads": 1, "grid_size": 1}),
        act=nn.ReLU,
        factorize=NMF,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.in_proj = Linear(in_channels, out_channels, bias=False)
        self.reshape = partialize(reshape)((None, out_channels, *spatial_size))
        self.act = partialize(act)()
        # reshaped_size = tuple(s for s in reshaped_size if s != 1)  # squeeze size
        self.reshaped_size = self.reshape.output_size[2:]
        self.factorize = partialize(factorize)(self.reshaped_size, **kwargs)
        self.out_proj = Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, S1, S2, ..., Sp)

        # input linear projection
        out = self.in_proj(x)

        # reshape (fold/unfold)
        out = self.reshape(out)

        # apply activation function
        out = self.act(out)

        # matrix or tensor factorization
        out = self.factorize(out)

        # reshape back (unfold/fold)
        out = self.reshape.inverse_forward(out)

        # output linear projection
        out = self.out_proj(out)

        # dropout
        out = self.dropout(out)
        return out


class FactorizerBlock(nn.Module):
    """Factorizer Block."""

    def __init__(
        self, channels, spatial_size, norm=LayerNorm, dropout=0.0, mlp_ratio=2, **kwargs
    ):
        super().__init__()

        self.norm1 = partialize(norm)(channels)
        self.fact = FactMixer(channels, channels, spatial_size, dropout=dropout, **kwargs)

        self.norm2 = partialize(norm)(channels)
        self.mlp = MLP(channels, ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        x = x + self.fact(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FactorizerStage(nn.Module):
    """Generic Factorizer block for one stage."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size,
        depth=1,
        adapter=(Linear, {"bias": False}),
        pos_embed=nn.Identity,
        dropout=0.0,
        **subblocks,
    ):
        super().__init__()
        if in_channels != out_channels:
            adapter = partialize(adapter)
            self.adapter = adapter(in_channels, out_channels)

        pos_embed = partialize(pos_embed)
        self.pos_embed = pos_embed(out_channels, spatial_size)
        if len(list(self.pos_embed.parameters())) > 0:
            self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                FactorizerBlock(
                    out_channels,
                    spatial_size,
                    **subblocks,
                )
            )

    def forward(self, x):
        # x: (B, C, S1, S2, ..., Sp)
        out = self.adapter(x) if hasattr(self, "adapter") else x
        out = self.pos_embed(out) if hasattr(self, "pos_embed") else out
        out = self.pos_drop(out) if hasattr(self, "pos_drop") else out
        for blk in self.blocks:
            out = blk(out)

        return out


class Factorizer(UNet):
    """Factorizer for Segmentation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        upsample=None,
        head=None,
        pos_embed=PositionalEmbedding,
        num_deep_supr=False,
        **kwargs,
    ):
        if stem is None:
            stem = (
                getattr(nn, f"Conv{len(spatial_size)}d"),
                {"kernel_size": 3, "padding": 1, "bias": False},
            )
        num_encoder_stages = len(encoder_depth)
        num_decoder_stages = len(decoder_depth)
        encoder_block = (num_encoder_stages - 1) * [(FactorizerStage, kwargs)]
        bottleneck_block = [(FactorizerStage, {"pos_embed": pos_embed, **kwargs})]
        decoder_block = num_decoder_stages * [(FactorizerStage, kwargs)]
        block = encoder_block + bottleneck_block + decoder_block
        super().__init__(
            in_channels,
            out_channels,
            spatial_dims=len(spatial_size),
            spatial_size=spatial_size,
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
        )
