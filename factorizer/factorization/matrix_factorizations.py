import math
import random

import torch
from torch import nn
import einops
import opt_einsum as oe

from ..utils.helpers import as_tuple, wrap_class, null_context
from .operations import t, dot, norm2, relative_error
from .base import MF


################################
# Initializers
################################


class RandomInit(nn.Module):
    def __init__(self, size, rank, method):
        super().__init__()
        self.method = method
        M, N = size
        self.register_buffer("u0", torch.empty(M, rank))
        self.register_buffer("v0", torch.empty(N, rank))
        init_method = getattr(nn.init, f"{method}_")
        init_method(self.u0)
        init_method(self.v0)
        self.flops = 0

    def forward(self, x):
        B = x.shape[0]
        u = einops.repeat(self.u0, "m r -> b m r", b=B)
        v = einops.repeat(self.v0, "n r -> b n r", b=B)
        return u, v


class FCMInit(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.fcm = FCM(size=size, rank=rank, num_iters=1)
        self.fcm.m = nn.Parameter(torch.tensor([2]))
        self.flops = self.fcm.flops["decompose"]

    def forward(self, x):
        u, v = self.fcm.decompose(x)
        return u, v


class EKMInit(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.ekm = EKM(size=size, rank=rank, num_iters=1)
        self.ekm.alpha = nn.Parameter(torch.tensor([0.1]))
        self.flops = self.ekm.flops["decompose"]

    def forward(self, x):
        u, v = self.ekm.decompose(x)
        return u, v


class SVDInit(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        (M, N), R = size, rank
        self.svd = SVD(size=size, rank=rank)
        self.flops = self.svd.flops["decompose"] + R + 2 * M * R + 2 * N * R

    def forward(self, x):
        u, s, v = self.svd.decompose(x)
        s = torch.sqrt(s)  # FLOPS = R
        u = torch.einsum("bir, br -> bir", u, s)  # FLOPS = 2MR
        v = torch.einsum("bjr, br -> bjr", v, s)  # FLOPS = 2NR
        return u, v


class NNDSVDInit(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.size = M, N = size
        self.rank = rank
        self.svd = SVD(size, rank)
        self.flops = self.svd.flops["decompose"] + 2 * (4 * rank * (M + N))

    def forward(self, x):
        u, s, v = self.svd.decompose(x)
        s = torch.sqrt(s)
        u = torch.einsum("bir, br -> bir", u, s)
        v = torch.einsum("bjr, br -> bjr", v, s)

        for r in range(self.rank):
            a = u[:, :, r].clone()
            b = v[:, :, r].clone()
            ap = torch.relu(a)
            an = torch.relu(-a)
            bp = torch.relu(b)
            bn = torch.relu(-b)
            abp = norm2(ap) * norm2(bp)
            abn = norm2(an) * norm2(bn)
            batch_mask = abp >= abn
            u[batch_mask, :, r] = ap[batch_mask, :].clone()
            v[batch_mask, :, r] = bp[batch_mask, :].clone()
            u[~batch_mask, :, r] = an[~batch_mask, :].clone()
            v[~batch_mask, :, r] = bn[~batch_mask, :].clone()

        return u, v


################################
# Solvers
################################


class AlternatingLeastSquares(nn.Module):
    "Alternating Least Squares."

    def __init__(self, factor=(0, 1), eps=1e-8, project=None, **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        self.eps = eps
        project = nn.Identity if project is None else project
        project = wrap_class(project)
        self.project = project()
        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        if None not in (self.size, self.rank):
            M, N = self.size
            R = self.rank
            self.flops = 0
            if 0 in self.factor:
                self.flops += math.ceil(
                    2 * N * (R ** 2) * M - (2 / 3) * (R ** 3) * M
                )

            if 1 in self.factor:
                self.flops += math.ceil(
                    2 * M * (R ** 2) * N - (2 / 3) * (R ** 3) * N
                )
        else:
            self.flops = None

    def single_update(self, x, u, v):
        # x ≈ u @ t(v) --> u = ?
        _, M, N = x.shape
        if M >= N:
            u_new = x @ t(torch.linalg.pinv(v))
        else:
            a, b = x @ v, t(v) @ v
            u_new = torch.linalg.solve(b, t(a))
            u_new = t(u_new)

        return self.project(u_new)

    def forward(self, x, factor_matrices):
        u, v = factor_matrices

        if 0 in self.factor:
            u = self.single_update(x, u, v)

        if 1 in self.factor:
            v = self.single_update(t(x), v, u)

        return u, v


class ProjectedGradient(nn.Module):
    "Projected gradient descent with line search for linear least squares."

    def __init__(self, factor=(0, 1), project=None, eps=1e-8, **kwargs):
        super().__init__()
        self.factor = factor
        self.eps = eps
        project = nn.Identity if project is None else project
        project = wrap_class(project)
        self.project = project()

        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        if None not in (self.size, self.rank):
            M, N = self.size
            R = self.rank
            self.flops = 0
            if 0 in self.factor:
                self.flops += 2 * M * N * R + 4 * M * R + 2 * N * R + 7 * M * R

            if 1 in self.factor:
                self.flops += 2 * N * M * R + 4 * N * R + 2 * M * R + 7 * N * R

        else:
            self.flops = None

    def single_update(self, x, u, v):
        # x ≈ u @ t(v)
        a, b = x @ v, t(v) @ v  # FLOPS = 2MNR + 2RNR
        g = a - u @ b  # gradient; FLOPS = MR + 2MRR
        η = (dot(g, g) + self.eps) / (
            dot(g, g @ b) + self.eps
        )  # FLOPS = 2MR + 2MR + 2MRR
        η = η.unsqueeze(-1)
        u_new = self.project(u + η * g)  # FLOPS = 2MR
        return u_new

    def forward(self, x, factor_matrices):
        u, v = factor_matrices

        if 0 in self.factor:
            u = self.single_update(x, u, v)

        if 1 in self.factor:
            v = self.single_update(t(x), v, u)

        return u, v


class CoordinateDescent(nn.Module):
    "Block coordinate descent update for linear least squares."

    def __init__(self, factor=(0, 1), eps=1e-8, project=None, **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        self.eps = eps
        project = nn.Identity if project is None else project
        project = wrap_class(project)
        self.project = project()

        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        if None not in (self.size, self.rank):
            M, N = self.size
            R = self.rank
            self.flops = 0
            if 0 in self.factor:
                self.flops += 2 * R * (M * N + N * R + M * R + M)

            if 1 in self.factor:
                self.flops += 2 * R * (N * M + M * R + N * R + N)
        else:
            self.flops = None

    def single_update(self, x, u, v):
        # x ≈ u @ t(v) --> u = ?
        R = u.shape[-1]
        a, b = x @ v, t(v) @ v  # FLOPS = 2MNR + 2RNR
        if R > 1:
            u_new = u.clone()
            for r in range(R):
                indices = [j for j in range(R) if j != r]
                term1 = a[:, :, r : (r + 1)].clone()
                term2 = u_new[:, :, indices].clone()
                term3 = b[:, indices, r : (r + 1)].clone()
                numerator = term1 - term2 @ term3 + self.eps  # FLOPS = 2MR
                denominator = b[:, r : (r + 1), r : (r + 1)].clone() + self.eps
                u_new[:, :, r : (r + 1)] = self.project(
                    numerator / denominator
                ).clone()  # FLOPS = 2M
        else:
            numerator = a + self.eps
            denominator = b + self.eps
            u_new = self.project(numerator / denominator)

        return u_new

    def forward(self, x, factor_matrices):
        u, v = factor_matrices

        if 0 in self.factor:
            u = self.single_update(x, u, v)

        if 1 in self.factor:
            v = self.single_update(t(x), v, u)

        return u, v


class MultiplicativeUpdate(nn.Module):
    def __init__(self, factor=(0, 1), eps=1e-8, **kwargs):
        super().__init__()
        self.factor = as_tuple(factor)
        self.eps = eps

        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        if None not in (self.size, self.rank):
            M, N = self.size
            R = self.rank
            self.flops = 0
            if 0 in self.factor:
                self.flops += 2 * R * (M * N + N * R + M * R + 2 * M)

            if 0 in self.factor:
                self.flops += 2 * R * (N * M + M * R + N * R + 2 * N)
        else:
            self.flops = None

    def single_update(self, x, u, v):
        # x ≈ u @ t(v)
        a, b = x @ v, t(v) @ v  # FLOPS = 2MNR + 2RNR
        numerator = u * a + self.eps  # FLOPS = 2MR
        denominator = u @ b + self.eps  # FLOPS = 2MRR + MR
        u_new = numerator / denominator  # FLOPS = MR
        return u_new

    def forward(self, x, factor_matrices):
        u, v = factor_matrices

        if 0 in self.factor:
            u = self.single_update(x, u, v)

        if 1 in self.factor:
            v = self.single_update(t(x), v, u)

        return u, v


class WeightedMultiplicativeUpdate(nn.Module):
    """Weighted multiplicative update for weighted NMF:

        min || W * (X - U t(V)) ||^2
        s.t. U, V ≥ 0
    """

    def __init__(self, factor=(0, 1), eps=1e-8, **kwargs):
        super().__init__()
        self.factor = factor
        self.eps = eps

        self.size = kwargs.get("size")  # size and rank used to count flops
        self.rank = kwargs.get("rank")
        if None not in (self.size, self.rank):
            M, N = self.size
            R = self.rank
            self.flops = 0
            if 0 in self.factor:
                self.flops += 2 * (3 * M * N * R + M * N + 2 * M * R)

            if 0 in self.factor:
                self.flops += 2 * (3 * N * M * R + N * M + 2 * N * R)
        else:
            self.flops = None

    def single_update(self, x, u, v, w):
        # x ≈ u @ t(v)
        a = (w * x) @ v  # FLOPS = MN + 2MNR
        numerator = u * a + self.eps  # FLOPS = 2MR
        denominator = (
            w * (u @ t(v))
        ) @ v + self.eps  # FLOPS = 2MRN + MN + 2MNR + MR
        u_new = numerator / denominator  # FLOPS = MR
        return u_new

    def forward(self, x, factor_matrices, weight=None):
        u, v = factor_matrices
        weight = torch.ones_like(x) if weight is None else weight

        if 0 in self.factor:
            u = self.single_update(x, u, v, weight)

        if 1 in self.factor:
            v = self.single_update(t(x), v, u, t(weight))

        return u, v


class FastMultiplicativeUpdate(nn.Module):
    def __init__(self, size, rank, factor=(0, 1), eps=1e-8, **kwargs):
        super().__init__()
        self.size = M, N = size
        self.rank = R = rank
        self.factor = as_tuple(factor)
        self.eps = eps

        x_size = (1, M, N)
        u_size = (1, M, R)
        v_size = (1, N, R)

        self.numerator_expr = {}
        self.denominator_expr = {}
        self.flops = 0
        if 0 in self.factor:
            numerator_eq = "bij, bir, bjr -> bir"
            _, con_info = oe.contract_path(
                numerator_eq,
                x_size,
                u_size,
                v_size,
                shapes=True,
                optimize="optimal",
            )
            expr = oe.contract_expression(
                numerator_eq, x_size, u_size, v_size, optimize=con_info.path
            )
            self.numerator_expr[0] = (expr, con_info)
            self.flops += con_info.opt_cost

            denominator_eq = "bis, bjs, bjr -> bir"
            _, con_info = oe.contract_path(
                denominator_eq,
                u_size,
                v_size,
                v_size,
                shapes=True,
                optimize="optimal",
            )
            expr = oe.contract_expression(
                denominator_eq, u_size, v_size, v_size, optimize=con_info.path
            )
            self.denominator_expr[0] = (expr, con_info)
            self.flops += con_info.opt_cost

        if 1 in self.factor:
            numerator_eq = "bij, bir, bjr -> bjr"
            _, con_info = oe.contract_path(
                numerator_eq,
                x_size,
                u_size,
                v_size,
                shapes=True,
                optimize="optimal",
            )
            expr = oe.contract_expression(
                numerator_eq, x_size, u_size, v_size, optimize=con_info.path
            )
            self.numerator_expr[1] = (expr, con_info)
            self.flops += con_info.opt_cost

            denominator_eq = "bir, bis, bjs -> bjr"
            _, con_info = oe.contract_path(
                denominator_eq,
                u_size,
                u_size,
                v_size,
                shapes=True,
                optimize="optimal",
            )
            expr = oe.contract_expression(
                denominator_eq, u_size, u_size, v_size, optimize=con_info.path
            )
            self.denominator_expr[1] = (expr, con_info)
            self.flops += con_info.opt_cost

    def forward(self, x, factor_matrices):
        u, v = factor_matrices

        if 0 in self.factor:
            numerator = self.numerator_expr[0][0](x, u, v) + self.eps
            denominator = self.denominator_expr[0][0](u, v, v) + self.eps
            u = numerator / denominator

        if 1 in self.factor:
            numerator = self.numerator_expr[1][0](x, u, v) + self.eps
            denominator = self.denominator_expr[1][0](u, u, v) + self.eps
            v = numerator / denominator

        return u, v


###################################
# K-means-based factorizers
###################################


class FCM(MF):
    """Fuzzy c-means (FCM), also called fuzzy k-means."""

    def __init__(
        self,
        size,
        rank=None,
        compression=None,
        m=2,
        num_iters=5,
        num_grad_steps=None,
        eps=1e-16,
        seed=42,
        verbose=False,
        **kwargs,
    ):
        self.m = m
        self.eps = eps
        self.seed = seed

        def init(size, rank):
            (M, _), R = size, rank

            def wrapper(x):
                random.seed(self.seed)
                inds = random.sample(range(M), R)
                v = t(x[:, inds, :])
                u = self.get_memberships(x, v)
                return u, v

            wrapper.flops = 0
            return wrapper

        def solver(size, rank):
            def wrapper(x, factors):
                _, v = factors
                u_new = self.get_memberships(x, v)
                v_new = self.get_centers(x, u_new)
                return u_new, v_new

            (M, N), R = size, rank
            wrapper.flops = 4 * M * R * N + 7 * M * R
            return wrapper

        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )

    def get_dist(self, x, v):
        x2 = (x ** 2).sum(2, keepdim=True)
        xv = x @ v
        v2 = (v ** 2).sum(1, keepdim=True)
        return torch.relu(x2 - 2 * xv + v2)

    def get_memberships(self, x, v):
        d = self.get_dist(x, v)  # FLOPS = 2MRN
        u = (d + self.eps) ** (1 / (1 - self.m))  # FLOPS = 2MR
        u = (u + self.eps) / (u.sum(2, keepdim=True) + self.eps)  # FLOPS = 2MR
        return u

    def get_centers(self, x, u):
        u = u ** self.m  # FLOPS = MR
        u = (u + self.eps) / (u.sum(1, keepdim=True) + self.eps)  # FLOPS = 2MR
        v = t(x) @ u  # FLOPS = 2MRN
        return v

    def loss(self, x, u, v):
        d = self.get_dist(x, v)
        return torch.mean((u ** self.m) * d)


class EKM(MF):
    """Entropy-based k-means: fuzzy k-means with etropy regularization."""

    def __init__(
        self,
        size,
        rank=None,
        compression=None,
        alpha=0.1,
        num_iters=5,
        num_grad_steps=None,
        eps=1e-16,
        seed=42,
        verbose=False,
        **kwargs,
    ):
        self.eps = eps
        self.seed = seed
        self.verbose = verbose

        def init(size, rank):
            (M, _), R = size, rank

            def wrapper(x):
                random.seed(self.seed)
                inds = random.sample(range(M), R)
                v = t(x[:, inds, :])
                u = self.get_memberships(x, v)
                return u, v

            wrapper.flops = 0
            return wrapper

        def solver(size, rank):
            def wrapper(x, factors):
                _, v = factors
                u_new = self.get_memberships(x, v)
                v_new = self.get_centers(x, u_new)
                return u_new, v_new

            (M, N), R = size, rank
            wrapper.flops = 4 * M * R * N + 7 * M * R
            return wrapper

        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )
        (_, N), R = self.size, self.rank
        self.alpha = alpha * N / (1 + math.log(R))

    def get_dist(self, x, v):
        x2 = (x ** 2).sum(2, keepdim=True)
        xv = x @ v
        v2 = (v ** 2).sum(1, keepdim=True)
        return torch.relu(x2 - 2 * xv + v2)

    def get_memberships(self, x, v):
        d = self.get_dist(x, v)  # FLOPS = 2MRN
        u = -d / self.alpha  # FLOPS = 2MR
        u = torch.softmax(u, dim=2)  # FLOPS = 3MR
        return u

    def get_centers(self, x, u):
        u = (u + self.eps) / (u.sum(1, keepdim=True) + self.eps)  # FLOPS = 2MR
        v = t(x) @ u  # FLOPS = 2MRN
        return v

    def loss(self, x, u, v):
        d = self.get_dist(x, v)
        h = torch.where(u > self.eps, u * u.log(), torch.zeros_like(u))
        h = h + (1 / self.rank) * math.log(self.rank)
        return torch.mean(u * d + self.alpha * h)


################################
# Low-rank matrix factorizers
################################


class SVD(nn.Module):
    """Singular Value Dcomposition."""

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        no_grad=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        self.size = M, N = size
        self.no_grad = no_grad

        assert (rank, compression) != (
            None,
            None,
        ), "'rank' or 'compression' must be specified."

        # degrees of freedom of the input matrix
        df_input = M * N
        # degrees of freedom of the low-rank matrix
        df_lowrank = M + N

        if rank is None:
            self.rank = R = rank = max(
                math.ceil(df_input / (compression * df_lowrank)), 1
            )
        else:
            self.rank = R = rank

        # update compression ratio
        self.compression = df_input / (rank * df_lowrank)

        # reconstruct expression
        reconstruct_eq = "bir, br, bjr -> bij"
        u_size, s_size, v_size = (1, M, R), (1, R), (1, N, R)
        _, self.reconstruct_info = oe.contract_path(
            reconstruct_eq,
            u_size,
            s_size,
            v_size,
            shapes=True,
            optimize="optimal",
        )
        self.reconstruct_expr = oe.contract_expression(
            reconstruct_eq,
            u_size,
            s_size,
            v_size,
            optimize=self.reconstruct_info.path,
        )

        # FLOPs
        self.flops = {"init": 0}
        self.flops["decompose"] = 2 * math.ceil(
            6 * M * N * R + (M + N) * (R ** 2)
        )
        self.flops["reconstruct"] = self.reconstruct_info.opt_cost

        self.verbose = verbose

    def context(self):
        # get context
        if self.no_grad:
            context = torch.no_grad()
        else:
            context = null_context()

        return context

    def decompose(self, x):
        # x: B × M × N
        with self.context():
            torch.manual_seed(42)
            u, s, v = torch.svd_lowrank(x, self.rank)

            if self.verbose:
                loss = self.loss(x, u, s, v)
                print(f"loss = {loss}")

        return u, s, v

    def reconstruct(self, u, s, v):
        return self.reconstruct_expr(u, s, v)

    def loss(self, x, u, s, v):
        return relative_error(x, self.reconstruct(u, s, v))

    def forward(self, x):
        u, s, v = self.decompose(x)
        return self.reconstruct(u, s, v)


class LRMA(MF):
    """Low-Rank Matrix Approximation.

        X ≈ U t(V),
        U, V ∈ S
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        num_iters=5,
        num_grad_steps=None,
        init="normal",
        solver="cd",
        verbose=False,
        **kwargs,
    ):
        # set factors initializer
        if f"{init}_" in dir(nn.init):
            init = (RandomInit, {"method": init})
        elif init == "fcm":
            init = FCMInit
        elif init == "ekm":
            init = EKMInit
        elif init == "svd":
            init = SVDInit
        else:
            init = init

        # set solver
        if solver == "cd":
            solver = CoordinateDescent
        elif solver == "gd":
            solver = ProjectedGradient
        elif solver == "als":
            solver = AlternatingLeastSquares
        else:
            solver = solver

        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )


class NMF(MF):
    """Nonnegative Matrix Factorization.

        X ≈ U t(V),
        X, U, V  >= 0
    """

    def __init__(
        self,
        size,
        rank=None,
        compression=10,
        num_iters=5,
        num_grad_steps=None,
        init="uniform",
        solver="hals",
        verbose=False,
        **kwargs,
    ):
        # set factors initializer
        if f"{init}_" in dir(nn.init):
            init = (RandomInit, {"method": init})
        elif init == "fcm":
            init = FCMInit
        elif init == "ekm":
            init = EKMInit
        elif init == "nndsvd":
            init = NNDSVDInit
        else:
            init = init

        # set solver
        if solver == "mu":
            solver = MultiplicativeUpdate
        elif solver == "wmu":
            solver = WeightedMultiplicativeUpdate
        elif solver in ("cd", "hals"):
            solver = (CoordinateDescent, {"project": nn.ReLU})
        elif solver == "gd":
            solver = (ProjectedGradient, {"project": nn.ReLU})
        elif solver == "als":
            solver = (AlternatingLeastSquares, {"project": nn.ReLU})
        else:
            solver = solver

        super().__init__(
            size,
            rank=rank,
            compression=compression,
            num_iters=num_iters,
            num_grad_steps=num_grad_steps,
            init=init,
            solver=solver,
            verbose=verbose,
            **kwargs,
        )

