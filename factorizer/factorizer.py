from torch import nn

from .utils.helpers import wrap_class
from .layers import LayerNorm, Linear, PositionalEmbedding
from .factorization import Matricize, NMF
from .unet import UNet


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FactorizerSubblock(nn.Module):
    """Generic Factorizer Sublock."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size,
        tensorize=(Matricize, {"num_heads": 1, "grid_size": 1}),
        act=nn.ReLU,
        factorize=NMF,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        tensorize = wrap_class(tensorize)
        act = wrap_class(act)
        factorize = wrap_class(factorize)

        self.in_proj = Linear(in_channels, out_channels, bias=False)

        self.tensorize = tensorize((None, out_channels, *spatial_size))

        self.act = act()

        tensorized_size = self.tensorize.output_size[2:]
        # tensorized_size = tuple(s for s in tensorized_size if s != 1)  # squeeze size
        self.tensorized_size = tensorized_size
        self.factorize = factorize(tensorized_size, **kwargs)

        self.out_proj = Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp

        # input linear projection
        out = self.in_proj(x)

        # tensorize (fold/unfold)
        out = self.tensorize(out)

        # apply activation function
        out = self.act(out)

        # # save the size
        # shape = out.shape

        # # remove singleton dims (squeeze)
        # out = out.reshape(*shape[:2], *self.tensorized_size)

        # matrix or tensor factorization
        out = self.factorize(out)

        # # back to the original size (unsqueeze)
        # out = out.reshape(shape)

        # detensorize (unfold/fold)
        out = self.tensorize.inverse_forward(out)

        # output linear projection
        out = self.out_proj(out)

        # dropout
        out = self.dropout(out)
        return out


class FactorizerBlock(nn.Module):
    """Generic Factorizer Block."""

    def __init__(
        self,
        channels,
        spatial_size,
        **subblocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleDict()
        for key, value in subblocks.items():
            subblock = wrap_class(value)
            self.blocks[key] = Residual(
                nn.Sequential(
                    LayerNorm(channels),
                    subblock(channels, channels, spatial_size=spatial_size),
                )
            )

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = x
        for blk in self.blocks.values():
            out = blk(out)

        return out


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
            adapter = wrap_class(adapter)
            self.adapter = adapter(in_channels, out_channels)

        pos_embed = wrap_class(pos_embed)
        self.pos_embed = pos_embed(out_channels, spatial_size)
        if len(list(self.pos_embed.parameters())) > 0:
            self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for j in range(depth):
            self.blocks.append(
                FactorizerBlock(
                    out_channels,
                    spatial_size,
                    **subblocks,
                )
            )

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = self.adapter(x) if hasattr(self, "adapter") else x
        out = self.pos_embed(out) if hasattr(self, "pos_embed") else out
        out = self.pos_drop(out) if hasattr(self, "pos_drop") else out
        for blk in self.blocks:
            out = blk(out)

        return out


class SegmentationFactorizer(UNet):
    """Factorizer for Segmentation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size=None,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        upsample=None,
        head=None,
        pos_embed=PositionalEmbedding,
        num_deep_supr=1,
        **kwargs,
    ):
        num_encoder_stages = len(encoder_depth)
        num_decoder_stages = len(decoder_depth)
        encoder_block = (num_encoder_stages - 1) * [FactorizerStage]
        bottleneck_block = [(FactorizerStage, {"pos_embed": pos_embed})]
        encoder_block = num_decoder_stages * [FactorizerStage]
        block = encoder_block + bottleneck_block + encoder_block
        super().__init__(
            in_channels,
            out_channels,
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
