import copy

import torch
from torch import nn
from torch.nn.modules.utils import _pair

from .utils.helpers import wrap_class, cumprod
from .factorization import Reshape, MLSVD
from .unet import UNet


###################################
# Factorizer Blocks
###################################


class LayerNorm(nn.Module):
    """"Layer norm for channels-first inputs."""

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim, **kwargs)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = torch.einsum("b c ... -> b ... c", x)
        out = self.norm(out)
        out = torch.einsum("b ... c -> b c ...", out)
        return out


class Linear(nn.Module):
    """"Linear layer for channels-first inputs."""

    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype=None,
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=2)
        self.linear = nn.Conv1d(
            in_features,
            out_features,
            kernel_size=1,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        shape = x.shape
        out = self.flatten(x)
        out = self.linear(out)
        out = out.reshape(shape[0], -1, *shape[2:])
        return out


class PosEmbed(nn.Module):
    """"Learnable positional embedding."""

    def __init__(self, input_dim, spatial_size, dropout=0.0, **kwargs):
        super().__init__()
        self.pos = nn.Parameter(torch.empty(1, input_dim, *spatial_size))
        nn.init.normal_(self.pos, std=1.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = x + self.pos
        out = self.dropout(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=None,
        ratio=2,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        hidden_dim = (
            int(ratio * input_dim) if hidden_dim is None else hidden_dim
        )
        dropout = _pair(dropout)

        self.block = nn.Sequential(
            Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout[0]),
            Linear(hidden_dim, output_dim),
            nn.Dropout(dropout[1]),
        )

    def forward(self, x):
        return self.block(x)


class DepthWiseP2P(nn.Module):
    "Depth-wise patch-to-patch transform."

    def __init__(self, size):
        super().__init__()
        # patches already flattened in the matricization step
        num_pixels = size[-1]  # last dim is #pixels in a patch
        self.linear = nn.Linear(num_pixels, num_pixels)

    def forward(self, x):
        # x: (batch × channels × patches) × pixels
        return self.linear(x)


class FactorizerSubblock(nn.Module):
    """Generic Factorizer Sublock."""

    def __init__(
        self,
        input_dim,
        output_dim,
        spatial_size,
        tensorize=Reshape,
        act=nn.GELU,
        factorize=MLSVD,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        tensorize = wrap_class(tensorize)
        act = wrap_class(act)
        factorize = wrap_class(factorize)

        self.linear = Linear(input_dim, output_dim, bias=False)
        self.tensorize = tensorize((None, output_dim, *spatial_size))
        self.act = act()

        tensorized_size = self.tensorize.output_size[1:]
        # squeeze size
        tensorized_size = tuple(s for s in tensorized_size if s != 1)
        self.tensorized_size = tensorized_size
        self.factorize = factorize(tensorized_size, **kwargs)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp

        # input linear projection
        out = self.linear(x)

        # tensorize (fold/unfold)
        out = self.tensorize(out)

        # apply activation function
        out = self.act(out)

        # save the size
        shape = out.shape

        # remove 1-dim modes (squeeze)
        out = out.reshape(shape[0], *self.tensorized_size)

        # tensor factorization)
        out = self.factorize(out)

        # back to the original size (unsqueeze)
        out = out.reshape(shape)

        # detensorize (unfold/fold)
        out = self.tensorize.inverse_forward(out)

        # dropout
        out = self.dropout(out)
        return out


class FactorizerBlock(nn.Module):
    """Generic Factorizer Block."""

    def __init__(
        self,
        input_dim,
        output_dim,
        spatial_size,
        adapter=None,
        pos_embed=False,
        dropout=0.0,
        **subblocks,
    ):
        super().__init__()
        if input_dim != output_dim:
            if adapter is None:
                self.adapter = nn.Sequential(
                    Linear(input_dim, output_dim, bias=False),
                    nn.Dropout(dropout),
                )
            else:
                adapter = wrap_class(adapter)
                self.adapter = adapter(input_dim, output_dim)

        if pos_embed:
            self.pos_embed = PosEmbed(input_dim, spatial_size, dropout=dropout)

        self.blocks = nn.ModuleDict()
        for key, value in subblocks.items():
            subblock = wrap_class(value)
            block = Residual(
                nn.Sequential(
                    LayerNorm(output_dim),
                    subblock(
                        output_dim, output_dim, spatial_size=spatial_size
                    ),
                )
            )
            self.blocks[key] = block

    def forward(self, x):
        # x: B × C × S1 × S2 × ... × Sp
        out = self.adapter(x) if hasattr(self, "adapter") else x
        out = self.pos_embed(out) if hasattr(self, "pos_embed") else out

        for block in self.blocks.values():
            out = block(out)

        return out


class SegmentationFactorizer(UNet):
    """Segmentation Factorizer (SeF)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        spatial_size,
        stem_width=None,
        encoder_depth=(1, 1, 1, 1, 1),
        encoder_width=(32, 64, 128, 256, 512),
        strides=(1, 2, 2, 2, 2),
        decoder_depth=(1, 1, 1, 1),
        stem=None,
        downsample=None,
        upsample=None,
        head=None,
        num_deep_supr=1,
        **kwargs,
    ):

        self.spatial_size = spatial_size
        spatial_dims = len(spatial_size)
        blocks = self._get_blocks(
            spatial_size, encoder_depth, decoder_depth, strides, **kwargs
        )
        super().__init__(
            in_channels,
            out_channels,
            spatial_dims=spatial_dims,
            stem_width=stem_width,
            encoder_depth=encoder_depth,
            encoder_width=encoder_width,
            strides=strides,
            decoder_depth=decoder_depth,
            stem=stem,
            downsample=downsample,
            block=blocks,
            upsample=upsample,
            head=head,
            num_deep_supr=num_deep_supr,
        )

    def _get_blocks(
        self, spatial_size, encoder_depth, decoder_depth, strides, **kwargs
    ):
        net_depth = encoder_depth + decoder_depth
        num_layers = len(net_depth)
        cumstrides = cumprod(strides)
        blocks = {}
        for layer, depth in enumerate(net_depth):
            for sublayer in range(depth):
                # get spatial size at current level
                level = min(layer, num_layers - 1 - layer)
                current_spatial_size = tuple(
                    s // cumstrides[level] for s in spatial_size
                )

                params = {
                    "spatial_size": current_spatial_size,
                    "pos_embed": (layer, sublayer) == (num_layers // 2, 0),
                    **kwargs,
                }
                blocks[(layer, sublayer)] = (FactorizerBlock, params)

        return blocks


def ablate(obj, name, indicator):
    class Ablated(obj):
        def _get_blocks(self, *args, **kwargs):
            blocks = super()._get_blocks(*args, **kwargs)
            for (layer, sublayer), (cl, params) in blocks.items():
                params_new = copy.deepcopy(params)
                if indicator(layer, sublayer):
                    params_new[name[0]][1][name[1]] = nn.Identity
                    blocks[(layer, sublayer)] = (cl, params_new)

            return blocks

    Ablated.__name__ = obj.__name__
    locals()[obj.__name__] = Ablated
    del Ablated
    return locals()[obj.__name__]
