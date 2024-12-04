from typing import Any, Optional, Callable, Sequence, ContextManager
import math
from contextlib import nullcontext

import torch
from torch import Tensor
from torch import nn
import opt_einsum as oe

from ..utils.helpers import as_tuple, partialize, is_partializable, PartialModuleType
from .operations import dot, norm2, relative_error


################################
# Initializers
################################


class Initializer(nn.Module):
    """Base class of an initializer for matrix factorization."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )


class RandomInit(Initializer):
    def __init__(
        self,
        rank: int,
        size: tuple[int, int],
        method: str | tuple[str, str] = "uniform",
    ) -> None:
        super().__init__()
        method = as_tuple(method)
        if len(method) == 1:
            self.method = (method[0], method[0])
        elif len(method) == 2:
            self.method = method
        else:
            raise ValueError("`method` not valid.")

        init_method = getattr(nn.init, f"{self.method[0]}_")
        self.register_buffer("u0", torch.empty(size[0], rank))
        init_method(self.u0)

        init_method = getattr(nn.init, f"{self.method[1]}_")
        self.register_buffer("v0", torch.empty(size[1], rank))
        init_method(self.v0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        u_shape = (*x.shape[:-2], *self.u0.shape)
        v_shape = (*x.shape[:-2], *self.v0.shape)

        u = self.u0.expand(u_shape)
        v = self.v0.expand(v_shape)
        return u, v


class SVDInit(Initializer):
    def __init__(self, size: tuple[int, int], rank: Optional[int] = None) -> None:
        super().__init__()
        self.svd = SVD(size=size, rank=rank)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        u, s, v = self.svd.decompose(x)
        s = torch.sqrt(s)
        u = torch.einsum("...ir, ...r -> ...ir", u, s)
        v = torch.einsum("...jr, ...r -> ...jr", v, s)
        return u, v


class NNDSVDInit(Initializer):
    def __init__(self, size: tuple[int, int], rank: Optional[int] = None) -> None:
        super().__init__()
        self.svd = SVD(size, rank)

    def forward(self, x) -> tuple[Tensor, Tensor]:
        u, s, v = self.svd.decompose(x)
        s = torch.sqrt(s)
        u = torch.einsum("bir, br -> bir", u, s)
        v = torch.einsum("bjr, br -> bjr", v, s)

        for r in range(self.svd.rank):
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


class BCDSolver(nn.Module):
    """Base class of a block coordinate descent solver for matrix factorization."""

    def __init__(self, factor: Sequence[int] = (0, 1), *args, **kwargs) -> None:
        super().__init__()
        self.factor = as_tuple(factor)
        assert set(self.factor).issubset({0, 1}), "`factor` elements must be 0 or 1."

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> u = ?
        raise NotImplementedError(
            f"Subclass {self.__class__.__name__} must implement this method."
        )

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> v = ?
        return self.update_u(x.mT, v, u)

    def forward(
        self, x: Tensor, factor_matrices: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        u, v = factor_matrices
        for j in self.factor:
            if j == 0:
                u = self.update_u(x, u, v)
            else:
                v = self.update_v(x, u, v)

        return u, v


class LeastSquares(BCDSolver):
    """Least Squares solver."""

    def __init__(
        self,
        factor: Sequence[int] = (0, 1),
        eps: float = 1e-16,
        project: Optional[PartialModuleType] = None,
        **kwargs,
    ):
        super().__init__(factor=factor)
        self.eps = eps
        project = nn.Identity if project is None else project
        project = partialize(project)
        self.project = project()

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> u = ?
        *_, M, N = x.shape
        if M >= N:
            u_new = x @ torch.linalg.pinv(v).mT
        else:
            a, b = x @ v, v.mT @ v
            u_new = torch.linalg.solve(b, a.mT)
            u_new = u_new.mT

        return self.project(u_new)


class ProjectedGradient(BCDSolver):
    "Projected gradient descent with line search for linear least squares."

    def __init__(
        self,
        factor: Sequence[int] = (0, 1),
        project: Optional[PartialModuleType] = None,
        eps: float = 1e-16,
        **kwargs,
    ):
        super().__init__(factor=factor)
        self.eps = eps
        project = nn.Identity if project is None else project
        project = partialize(project)
        self.project = project()

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> u = ?
        a, b = x @ v, v.mT @ v
        g = a - u @ b
        η = (dot(g, g) + self.eps) / (dot(g, g @ b) + self.eps)
        η = η.unsqueeze(-1)
        u_new = self.project(u + η * g)
        return u_new


class CoordinateDescent(BCDSolver):
    "Block coordinate descent update for linear least squares."

    def __init__(
        self,
        factor: Sequence[int] = (0, 1),
        eps: float = 1e-16,
        project: Optional[PartialModuleType] = None,
        **kwargs,
    ):
        super().__init__(factor=factor)
        self.eps = eps
        project = nn.Identity if project is None else project
        project = partialize(project)
        self.project = project()

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> u = ?
        R = u.shape[-1]
        a, b = x @ v, v.mT @ v
        if R > 1:
            u_new = u.clone()
            for r in range(R):
                indices = [j for j in range(R) if j != r]
                term1 = a[..., r : (r + 1)].clone()
                term2 = u_new[..., indices].clone()
                term3 = b[..., indices, r : (r + 1)].clone()
                numerator = term1 - term2 @ term3 + self.eps
                denominator = b[..., r : (r + 1), r : (r + 1)].clone() + self.eps
                u_new[..., r : (r + 1)] = self.project(numerator / denominator).clone()
        else:
            numerator = a + self.eps
            denominator = b + self.eps
            u_new = self.project(numerator / denominator)

        return u_new


class MultiplicativeUpdate(BCDSolver):
    "Multiplicative update for nonnegative matrix factorization (NMF)."

    def __init__(
        self, factor: Sequence[int] = (0, 1), eps: float = 1e-16, **kwargs
    ) -> None:
        super().__init__(factor=factor)
        self.eps = eps

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> u = ?
        a, b = x @ v, v.mT @ v
        numerator = u * a + self.eps
        denominator = u @ b + self.eps
        u_new = numerator / denominator
        return u_new


class FastMultiplicativeUpdate(BCDSolver):
    """Fast multiplicative update for NMF."""

    def __init__(
        self,
        factor: Sequence[int] = (0, 1),
        eps: float = 1e-16,
        **kwargs,
    ):
        super().__init__(factor=factor)
        self.eps = eps

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> u = ?
        numerator = torch.einsum("...ij, ...ir, ...jr -> ...ir", x, u, v) + self.eps
        denominator = torch.einsum("...is, ...js, ...jr -> ...ir", u, v, v) + self.eps
        u_new = numerator / denominator
        return u_new

    def update_v(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> v = ?
        numerator = torch.einsum("...ij, ...ir, ...jr -> ...jr", x, u, v) + self.eps
        denominator = torch.einsum("...ir, ...is, ...js -> ...jr", u, u, v) + self.eps
        v_new = numerator / denominator
        return v_new


class WeightedMultiplicativeUpdate(BCDSolver):
    """Multiplicative update for weighted NMF, i.e,

    min || W * (X - U @ V.mT) ||^2
    s.t. U, V ≥ 0
    """

    def __init__(
        self, factor: Sequence[int] = (0, 1), eps: float = 1e-16, **kwargs
    ) -> None:
        super().__init__(factor=factor)
        self.eps = eps

    def update_u(self, x: Tensor, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> u = ?
        a = (w * x) @ v
        numerator = u * a + self.eps
        denominator = (w * (u @ v.mT)) @ v + self.eps
        u_new = numerator / denominator
        return u_new

    def update_v(self, x: Tensor, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> v = ?
        return self.update_u(x.mT, v, u, w.mT)

    def forward(
        self,
        x: Tensor,
        factor_matrices: tuple[Tensor, Tensor],
        w: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        u, v = factor_matrices
        w = torch.ones_like(x) if w is None else w
        for j in self.factor:
            if j == 0:
                u = self.update_u(x, u, v, w)
            else:
                v = self.update_v(x, u, v, w)

        return u, v


class SemiMultiplicativeUpdate(BCDSolver):
    """Multiplicative update for Semi-NMF:

    min || X - U @ V.mT) ||^2
    s.t. U ≥ 0
    """

    def __init__(
        self,
        factor: Sequence[int] = (0, 1),
        eps: float = 1e-16,
        **kwargs,
    ) -> None:
        super().__init__(factor=factor)
        self.eps = eps

    def update_u(self, x: Tensor, u: Tensor, v: Tensor) -> Tensor:
        # x ≈ u @ v.mT --> u = ?
        a, b = x @ v, v.mT @ v
        numerator = torch.relu(a) + u @ torch.relu(-b) + self.eps
        denominator = torch.relu(-a) + u @ torch.relu(b) + self.eps
        u_new = u * torch.sqrt(numerator / denominator)
        return u_new


class Compose(BCDSolver, Sequence):
    def __init__(self, solvers: Optional[Sequence[BCDSolver]] = None, **kwargs) -> None:
        super().__init__()
        if solvers is None:
            solvers = []

        solvers = as_tuple(solvers)
        self.solvers = []
        self.factor = []
        self.size = kwargs.get("size")
        self.rank = kwargs.get("rank")

        for solver in solvers:
            solver = partialize(solver)
            solver = solver(**kwargs)
            self.solvers.append(solver)
            self.factor.append(getattr(solver, "factor"))

    def forward(
        self, x: Tensor, factor_matrices: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        u, v = factor_matrices
        for solver in self.solvers:
            u, v = solver(x, (u, v))

        return u, v

    def __setitem__(self, idx: int, solver: BCDSolver) -> None:
        self.solvers[idx] = solver

    def __getitem__(self, idx: int) -> BCDSolver:
        return self.solvers[idx]

    def __len__(self) -> int:
        return len(self.solvers)


###################################
# Matrix Factorization
###################################


class SVD(nn.Module):
    """Singular Value Decomposition."""

    def __init__(
        self,
        size: tuple[int, int],
        rank: Optional[int] = None,
        compression: float = 10,
        no_grad: bool = False,
        verbose: bool = False,
    ) -> None:
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
        self.verbose = verbose

    def context(self) -> ContextManager[None]:
        # get context
        if self.no_grad:
            context = torch.no_grad()
        else:
            context = nullcontext()

        return context

    def decompose(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # x: (B, M, N)
        with self.context():
            torch.manual_seed(42)
            u, s, v = torch.svd_lowrank(x, self.rank)

            if self.verbose:
                loss = self.loss(x, u, s, v)
                print(f"loss = {loss}")

        return u, s, v

    def reconstruct(self, u: Tensor, s: Tensor, v: Tensor) -> Tensor:
        return torch.einsum("...ir, ...r, ...jr -> ...ij", u, s, v)

    def loss(self, x: Tensor, u: Tensor, s: Tensor, v: Tensor) -> Tensor:
        return relative_error(x, self.reconstruct(u, s, v))

    def forward(self, x: Tensor) -> Tensor:
        u, s, v = self.decompose(x)
        return self.reconstruct(u, s, v)


class MatrixFactorization(nn.Module):
    """Base module for matrix factorization.

    X ≈ U @ V.mT,
    U, V ∈ S
    """

    def __init__(
        self,
        size: Sequence[int],
        rank: Optional[int] = None,
        compression: float = 10,
        init: str | PartialModuleType = "normal",
        solver: (str | PartialModuleType) | Sequence[str | PartialModuleType] = "cd",
        num_iters: int = 5,
        num_grad_steps: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.size = M, N = size
        self.num_iters = num_iters
        self.num_grad_steps = num_iters if num_grad_steps is None else num_grad_steps

        assert (rank, compression) != (
            None,
            None,
        ), "'rank' or 'compression' must be specified."

        # degrees of freedom of the input matrix
        df_input = M * N
        # degrees of freedom of the low-rank matrix
        df_lowrank = M + N

        if rank is None:
            self.rank = rank = max(math.ceil(df_input / (compression * df_lowrank)), 1)
        else:
            self.rank = rank

        # update compression ratio
        self.compression = df_input / (self.rank * df_lowrank)

        # set factors initializer
        init = partialize(_parse_init(init))
        self.init = init(size=size, rank=rank)

        # set solver
        solver = partialize(_parse_solver(solver))
        self.solver = solver(size=size, rank=rank)

        self.verbose = verbose

    def context(self, it: int) -> ContextManager[None]:
        if it < self.num_iters - self.num_grad_steps + 1:
            context = torch.no_grad()
        else:
            context = nullcontext()

        return context

    def decompose(self, x: Tensor, *args, **kwargs) -> tuple[Tensor, Tensor]:
        # x: (B, M, N)

        # initialize
        with self.context(0):
            u, v = self.init(x)

        # iterate
        for it in range(1, self.num_iters + 1):
            with self.context(it):
                if self.verbose:
                    loss = self.loss(x, u, v)
                    print(f"iter {it}, loss = {loss}")

                u, v = self.solver(x, [u, v], *args, **kwargs)

        return u, v

    def reconstruct(self, u: Tensor, v: Tensor) -> Tensor:
        return u @ v.mT

    def loss(
        self,
        x: Tensor,
        u: Tensor,
        v: Tensor,
        w: Optional[Tensor] = None,
    ) -> Tensor:
        return relative_error(x, self.reconstruct(u, v), w)

    def forward(self, x: Tensor) -> Tensor:
        u, v = self.decompose(x)
        return self.reconstruct(u, v)


class NMF(MatrixFactorization):
    """Nonnegative Matrix Factorization.

    X ≈ U @ V.mT,
    X, U, V  >= 0
    """

    def __init__(
        self,
        size: tuple[int, int],
        rank: Optional[int] = None,
        compression: float = 10,
        num_iters: int = 5,
        num_grad_steps: Optional[int] = None,
        init: str | PartialModuleType = "uniform",
        solver: (str | PartialModuleType) | Sequence[str | PartialModuleType] = "hals",
        verbose: bool = False,
        **kwargs,
    ):
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


INIT_DISPATCH_MAP = {
    "uniform": (RandomInit, {"method": "uniform"}),
    "normal": (RandomInit, {"method": "normal"}),
    "normal-uniform": (RandomInit, {"method": ("normal", "uniform")}),
    "uniform-normal": (RandomInit, {"method": ("uniform", "normal")}),
    "svd": SVDInit,
    "nndsvd": NNDSVDInit,
}

SOLVER_DISPATCH_MAP = {
    "mu": MultiplicativeUpdate,
    "mu-0": (MultiplicativeUpdate, {"factor": 0}),
    "mu-1": (MultiplicativeUpdate, {"factor": 1}),
    "fmu": FastMultiplicativeUpdate,
    "fmu-0": (FastMultiplicativeUpdate, {"factor": 0}),
    "fmu-1": (FastMultiplicativeUpdate, {"factor": 1}),
    "wmu": WeightedMultiplicativeUpdate,
    "wmu-0": (MultiplicativeUpdate, {"factor": 0}),
    "wmu-1": (MultiplicativeUpdate, {"factor": 1}),
    "smu": SemiMultiplicativeUpdate,
    "smu-0": (SemiMultiplicativeUpdate, {"factor": 0}),
    "smu-1": (SemiMultiplicativeUpdate, {"factor": 1}),
    "cd": CoordinateDescent,
    "cd-0": (CoordinateDescent, {"factor": 0}),
    "cd-1": (CoordinateDescent, {"factor": 1}),
    "nncd": (CoordinateDescent, {"project": nn.ReLU}),
    "nncd-0": (CoordinateDescent, {"factor": 0, "project": nn.ReLU}),
    "nncd-1": (CoordinateDescent, {"factor": 1, "project": nn.ReLU}),
    "hals": (CoordinateDescent, {"project": nn.ReLU}),
    "hals-0": (CoordinateDescent, {"factor": 0, "project": nn.ReLU}),
    "hals-1": (CoordinateDescent, {"factor": 1, "project": nn.ReLU}),
    "ls": LeastSquares,
    "ls-0": (LeastSquares, {"factor": 0}),
    "ls-1": (LeastSquares, {"factor": 1}),
    "nnls": (LeastSquares, {"project": nn.ReLU}),
    "nnls-0": (LeastSquares, {"factor": 0, "project": nn.ReLU}),
    "nnls-1": (LeastSquares, {"factor": 1, "project": nn.ReLU}),
}


def _dispatch(obj: Any, dispatch_map: dict[str, Any]) -> Any:
    """
    Dispatches an object to the appropriate value from a given dispatch map.

    Args:
        obj (Any): The object to dispatch.
        dispatch_map (dict[str, Any]): A dictionary mapping strings to objects.

    Returns:
        Any: The dispatched object if `obj` is a string, otherwise `obj` as-is.
    """
    out = dispatch_map.get(obj, obj) if isinstance(obj, str) else obj
    return out


def _parse_init(obj: str | PartialModuleType) -> PartialModuleType:
    """
    Parses an initializer object or string into a partial module type.

    Args:
        obj (str | PartialModuleType): An object or string to parse.

    Returns:
        PartialModuleType: The parsed partial module type.
    """
    return _dispatch(obj, INIT_DISPATCH_MAP)


def _parse_solver(
    obj: (str | PartialModuleType) | Sequence[str | PartialModuleType],
) -> PartialModuleType:
    """
    Parses a solver object, string, or sequence into a partial module type.

    This function checks if the input object is partializable or a string and
    dispatches it accordingly using a solver dispatch map. If the input is a
    sequence, each element is processed similarly, and a composed solver is
    returned.

    Args:
        obj ((str | PartialModuleType) | Sequence[str | PartialModuleType]):
            A solver object, string, or sequence of these to parse.

    Returns:
        PartialModuleType: The parsed partial module type.

    Raises:
        ValueError: If the input cannot be parsed into a valid solver type.
    """
    if is_partializable(obj):
        return obj
    elif isinstance(obj, str):
        return _dispatch(obj, SOLVER_DISPATCH_MAP)
    elif isinstance(obj, Sequence):
        out = []
        for x in obj:
            if is_partializable(x):
                out.append(x)
            elif isinstance(x, str):
                out.append(_dispatch(x, SOLVER_DISPATCH_MAP))
            else:
                raise ValueError
        return (Compose, {"solvers": out})
    else:
        raise ValueError
