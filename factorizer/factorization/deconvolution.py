from typing import Any, Optional, Sequence
import math
from functools import partial
from contextlib import nullcontext

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from torch.func import vmap
from einops.layers.torch import Rearrange

from .operations import relative_error
from ..layers import Linear


CONV = {d: getattr(F, f"conv{d}d") for d in range(1, 4)}


def conv(input: Tensor, weight: Tensor, groups: int = 1, **kwargs) -> Tensor:
    batch_size = input.shape[0]
    spatial_dims = input.ndim - 2

    # Reshape input and weight for batch computation
    input_reshaped = input.reshape(1, batch_size * input.shape[1], *input.shape[2:])
    weight_reshaped = weight.reshape(
        batch_size * weight.shape[1], weight.shape[2], *weight.shape[3:]
    )

    # Adjust groups for batch computation
    groups = groups * batch_size

    # Perform convolution
    output = CONV[spatial_dims](input_reshaped, weight_reshaped, groups=groups, **kwargs)

    # Reshape output back to batch form
    output = output.reshape(batch_size, -1, *output.shape[2:])

    return output


@vmap
def sconv(input1: Tensor, input2: Tensor, **kwargs) -> Tensor:
    spatial_dims = input1.ndim - 1
    input1 = input1.unsqueeze(1)
    input2 = input2.unsqueeze(1)
    output = CONV[spatial_dims](input1, input2, **kwargs)
    return output


def t(x: Tensor) -> Tensor:
    return x.transpose(1, 2)


def flip(h: Tensor) -> Tensor:
    return torch.flip(h, dims=tuple(range(3 - h.ndim, 0)))


class Initializer(nn.Module):
    def __init__(
        self,
        channels: int,
        source_channels: int,
        kernel_size: Sequence[int],
        groups: int,
    ) -> None:
        super().__init__()
        groups = channels if groups is None else groups
        assert channels % groups == 0, "`channels` must be divisible by groups"

        # Initialize h0 parameter
        h0 = torch.empty(channels, source_channels, *kernel_size)
        nn.init.kaiming_uniform_(h0, a=math.sqrt(5))
        self.h0 = nn.Parameter(h0)

        # Initialize linear layer
        self.linear = Linear(channels, groups * source_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        h_shape = (batch, *self.h0.shape)
        h = self.h0.expand(h_shape)
        s = self.linear(x)
        return F.relu(s), F.relu(h)


class Deconv(nn.Module):
    """Deconvolution layer."""

    def __init__(
        self,
        channels: int,
        kernel_size=Sequence[int],
        source_channels: Optional[int] = None,
        ratio: float = 4,
        groups: int = 8,
        update_source=True,
        update_filter=False,
        eps: float = 1e-16,
        num_iters: int = 2,
        num_grad_iters: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.groups = channels if groups == -1 else groups
        assert self.channels % self.groups == 0, "`channels` must be divisible by groups"
        self.source_channels = round(
            channels * ratio / self.groups if source_channels is None else source_channels
        )
        self.kernel_size = kernel_size
        self.init = Initializer(
            self.channels, self.source_channels, self.kernel_size, self.groups
        )
        self.update_source = update_source
        self.update_filter = update_filter
        self.num_iters = num_iters
        self.num_grad_iters = num_iters if num_grad_iters is None else num_grad_iters
        self.eps = eps
        self.verbose = verbose

        self.split_channels = Rearrange("b (g c) ... -> (b g) c ...", g=self.groups)
        self.merge_channels = Rearrange("(b g) c ... -> b (g c) ...", g=self.groups)
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = partial(conv, padding=padding, **kwargs)
        self.sconv = partial(sconv, padding=padding, **kwargs)

    def normalize_h(self, h: Tensor) -> Tensor:
        return (h + self.eps) / (
            h.sum([d for d in range(h.ndim) if d not in (0, 2)], keepdim=True) + self.eps
        )

    def update_s(self, x: Tensor, s: Tensor, h: Tensor) -> Tensor:
        # x ≈ conv(s,h) --> s = ?
        numerator = self.conv(x, t(flip(h))) + self.eps
        denominator = self.conv(self.conv(s, h), t(flip(h))) + self.eps
        return s * numerator / denominator

    def update_h(self, x: Tensor, s: Tensor, h: Tensor) -> Tensor:
        # x ≈ conv(s,h) --> h = ?
        numerator = self.sconv(s, x) + self.eps
        denominator = self.sconv(s, self.conv(s, h)) + self.eps
        return h * t(numerator / denominator)

    def update(self, x: Tensor, s: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        if self.update_source:
            s = self.update_s(x, s, h)

        if self.update_filter:
            h = self.update_h(x, s, h)

        return s, h

    def context(self, it: int) -> Any:
        if it < self.num_iters - self.num_grad_iters + 1:
            context = torch.no_grad()
        else:
            context = nullcontext()

        return context

    def iterative_update(self, x: Tensor, s: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        for it in range(1, self.num_iters + 1):
            with self.context(it):
                if self.verbose:
                    loss = self.loss(x, s, h)
                    print(f"iter {it}: loss = {loss}")

                s, h = self.update(x, s, h)

        return s, h

    def fit(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: (B, C, ...)

        # Initialize source and filter tensors
        s, h = self.init(x)

        # Split channels if grouping is enabled
        if self.groups != 1:
            x = self.split_channels(x)
            s = self.split_channels(s)
            h = self.split_channels(h)

        # Perform iterative update on source and filter tensors
        s, h = self.iterative_update(x, s, h)

        # Merge channels back if they were split
        if self.groups != 1:
            s = self.merge_channels(s)
            h = self.merge_channels(h)

        return s, h

    def reconstruct(self, s: Tensor, h: Tensor) -> Tensor:
        # Split channels if grouping is enabled
        if self.groups != 1:
            s = self.split_channels(s)
            h = self.split_channels(h)

        # Compute the reconstructed input tensor
        x_hat = self.conv(s, h)

        # Merge channels back if they were split
        if self.groups != 1:
            x_hat = self.merge_channels(x_hat)

        return x_hat

    def loss(
        self,
        x: Tensor,
        s: Tensor,
        h: Tensor,
    ) -> Tensor:
        return relative_error(x, self.conv(s, h))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, ...)

        # Initialize source and filter tensors
        s, h = self.init(x)

        # Split channels if grouping is enabled
        if self.groups != 1:
            x = self.split_channels(x)
            s = self.split_channels(s)
            h = self.split_channels(h)

        # Perform iterative update on source and filter tensors
        s, h = self.iterative_update(x, s, h)

        # Merge channels back if they were split
        if self.groups != 1:
            s = self.merge_channels(s)

        return s
