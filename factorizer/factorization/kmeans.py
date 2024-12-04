from typing import Optional, ContextManager
import math
import random
from contextlib import nullcontext

import torch
from torch import Tensor
from torch import nn


class KMeans(nn.Module):
    """A PyTorch module for k-means clustering."""

    def __init__(
        self,
        num_centers: int,
        num_iters: int = 10,
        num_grad_steps: Optional[int] = None,
        eps: float = 1e-16,
        seed: int = 42,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_centers = num_centers
        self.num_iters = num_iters
        self.num_grad_steps = num_iters if num_grad_steps is None else num_grad_steps
        self.eps = eps
        self.seed = seed
        self.verbose = verbose

    @staticmethod
    def get_dist(x: Tensor, v: Tensor) -> Tensor:
        """Compute squared Euclidean distances between `x` and `v`."""
        x2 = (x**2).sum(-1, keepdim=True)
        xv = x @ v.mT
        v2 = (v.mT**2).sum(-2, keepdim=True)
        return torch.relu(x2 - 2 * xv + v2)

    def get_clusters(self, x: Tensor, v: Tensor) -> Tensor:
        d = self.get_dist(x, v)
        return torch.argmin(d, dim=-1, keepdim=True)

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Update memberships."""
        clusters = self.get_clusters(x, v)  # Shape: (..., M, 1)
        u_new = torch.zeros(*x.shape[:-1], self.num_centers, device=x.device)
        u_new.scatter_(dim=-1, index=clusters, value=1.0)
        return u_new

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Update centroids."""
        u = (u + self.eps) / (u.sum(1, keepdim=True) + self.eps)
        v_new = u.mT @ x
        return v_new

    def update(self, x: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        u = self.update_u(x, u, v)
        v = self.update_v(x, u, v)
        return u, v

    def context(self, it: int) -> ContextManager[None]:
        if it < self.num_iters - self.num_grad_steps + 1:
            context = torch.no_grad()
        else:
            context = nullcontext()

        return context

    def initialize(self, x: Tensor) -> tuple[Tensor, Tensor]:
        random.seed(self.seed)
        inds = random.sample(range(x.shape[-2]), self.num_centers)
        v = x[..., inds, :]
        u = self.update_u(x, None, v)
        return u, v

    def loss(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        d = self.get_dist(x, v)
        d_avg = (d * u).sum(dim=(-2, -1))
        d_avg = d_avg / (u.shape[-2] * u.shape[-1])
        return d_avg

    def forward(self, x: Tensor, *args, **kwargs) -> tuple[Tensor, Tensor]:
        # x: (..., M, N)

        # initialize
        with self.context(0):
            u, v = self.initialize(x)

        # iterate
        for it in range(1, self.num_iters + 1):
            with self.context(it):
                if self.verbose:
                    loss = self.loss(x, u, v)
                    print(f"iter {it}, loss = {loss}")

                u, v = self.update(x, u, v)

        return u, v


class FuzzyCMeans(KMeans):
    """Fuzzy c-means (FCM)."""

    def __init__(self, m: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.m = m

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        d = self.get_dist(x, v)
        u = (d + self.eps) ** (1 / (1 - self.m))
        u = (u + self.eps) / (u.sum(-1, keepdim=True) + self.eps)
        u_new = u**self.m
        return u_new


class EntropyKMeans(KMeans):
    """Entropy k-means (EKM)."""

    def __init__(self, alpha: int = 0.001, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        d = self.get_dist(x, v)
        u_new = torch.softmax(-d / self.alpha, dim=2)
        return u_new

    def loss(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        d = self.get_dist(x, v)
        h = torch.where(u > self.eps, u * u.log(), torch.zeros_like(u))
        h = h + (1 / self.num_centers) * math.log(self.num_centers)
        loss = u * d + self.alpha * h
        loss_avg = loss.sum(dim=(-2, -1)) / (u.shape[-2] * u.shape[-1])
        return loss_avg
