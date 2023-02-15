import torch
from torch import nn
from torchvision.ops import StochasticDepth

from .linear import Linear
from ..utils.helpers import as_tuple, prod, wrap_class


class DoubleConv(nn.Module):
    """(Conv -- Drop -- Norm -- Act) ** 2."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.block1 = nn.Sequential(
            conv(in_channels, mid_channels, stride=stride),
            drop(),
            norm(mid_channels),
            act(),
        )

        self.block2 = nn.Sequential(
            conv(mid_channels, out_channels, stride=1),
            drop(),
            norm(out_channels),
            act(),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv1 = wrap_class(conv)
        conv2 = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.conv1 = conv1(in_channels, mid_channels, stride=stride)
        self.drop1 = drop()
        self.norm1 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, out_channels)
        self.drop2 = drop()
        self.norm2 = norm(out_channels)
        self.act = act()

        if prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            self.shortcut = conv[0](
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.norm2(out)

        out += shortcut

        out = self.act(out)

        return out


class BottleneckBlock(nn.Module):
    """ResNet bottleneck block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        alpha=0.5,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = int(in_channels * alpha)

        conv2 = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.conv1 = conv[0](in_channels, mid_channels, kernel_size=1)
        self.drop1 = drop()
        self.norm1 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, mid_channels, stride=stride)
        self.drop2 = drop()
        self.norm2 = norm(mid_channels)
        self.conv3 = conv[0](mid_channels, out_channels, kernel_size=1)
        self.drop3 = drop()
        self.norm3 = norm(mid_channels)
        self.act = act()

        if prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            self.shortcut = conv[0](
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        out = self.conv1(x)
        out = self.drop1(out)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.drop2(out)
        out = self.norm2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.drop3(out)
        out = self.norm3(out)

        out += shortcut
        out = self.act(out)

        return out


class PreActivationBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        conv=(nn.Conv3d, {"kernel_size": 3, "padding": 1}),
        norm=(nn.GroupNorm, (8,)),
        act=nn.LeakyReLU,
        drop=(nn.Dropout, {"p": 0.0}),
        stride=1,
        **kwargs,
    ):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels

        conv1 = wrap_class(conv)
        conv2 = wrap_class(conv)
        drop = wrap_class(drop)
        norm = wrap_class(norm)
        act = wrap_class(act)

        self.norm1 = norm(in_channels)
        self.act = act()
        self.conv1 = conv1(in_channels, mid_channels, stride=stride)
        self.drop1 = drop()
        self.norm2 = norm(mid_channels)
        self.conv2 = conv2(mid_channels, out_channels)
        self.drop2 = drop()

        if prod(as_tuple(stride)) != 1 or in_channels != out_channels:
            self.shortcut = conv[0](
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        out = self.norm1(x)
        out = self.act(out)
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.drop1(out)

        out = self.norm2(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.drop2(out)

        out += shortcut
        return out


class CovNeXtBlock(nn.Module):
    """ConvNeXt Block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mlp_ratio=4,
        stochastic_depth_prob=0.0,
        layer_scale=1e-6,
        conv=(nn.Conv3d, {"kernel_size": 7, "padding": 3}),
        norm=nn.LayerNorm,
        act=nn.GELU,
        **kwargs,
    ):
        super().__init__()

        conv = wrap_class(conv)
        norm = wrap_class(norm)
        act = wrap_class(act)

        if in_channels != out_channels:
            self.adapter = Linear(in_channels, out_channels, bias=False)

        hidden_channels = int(mlp_ratio * in_channels)
        self.dwconv = conv(out_channels, out_channels, groups=out_channels)
        self.norm = norm(out_channels)
        self.pwconv1 = nn.Linear(out_channels, hidden_channels)
        self.act = act()
        self.pwconv2 = nn.Linear(hidden_channels, out_channels)
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones(out_channels))
            if layer_scale > 0
            else None
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, x):
        x = self.adapter(x) if hasattr(self, "adapter") else x
        out = self.dwconv(x)
        out = torch.einsum("b c ... -> b ... c", out)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        if self.gamma is not None:
            out = self.gamma * out
        out = torch.einsum("b ... c -> b c ...", out)
        out = self.stochastic_depth(out)
        out += x
        return x


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        ratio=2,
        spatial_dims=3,
        act=nn.GELU,
        kernel_size=5,
        stride=1,
        padding=2,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden_channels = (
            int(ratio * in_channels) if hidden_channels is None else hidden_channels
        )
        conv = getattr(nn, f"Conv{spatial_dims}d")
        act = wrap_class(act)

        self.pwconv1 = Linear(in_channels, hidden_channels, bias=False)
        self.act = act()
        self.dwconv = conv(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            groups=hidden_channels,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.pwconv2 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        out = self.pwconv1(x)
        out = self.act(out)
        out = self.dwconv(out)
        out = self.pwconv2(out)
        return out
