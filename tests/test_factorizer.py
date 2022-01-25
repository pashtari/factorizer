# %%
import sys
import torch
from torch import nn

import factorizer as ft
from factorizer.factorization.tensor_factorizations import RandomTensorInit

# # %% FCM
# torch.manual_seed(42)
# size = (8, 8, 8, 8)
# x = torch.randn(1, *size, requires_grad=True)
# x = x.flatten(2, -1).transpose(1, 2)
# _, *size = x.shape

# fcm = ft.FCM(
#     size=size,
#     compression=10,
#     m=2,
#     num_iters=10,
#     num_grad_steps=None,
#     seed=42,
#     verbose=True,
# )
# u, v = fcm.decompose(x)
# print(f"FCM: FLOPS = {fcm.flops}")
# print(f"FCM: #nonzeros = {(u==0.0).sum()}")
# loss = u.sum()
# loss.backward()

# # %% EKM
# torch.manual_seed(42)
# size = (8, 8, 8, 8)
# x = torch.randn(1, *size, requires_grad=True)
# x = x.flatten(2, -1).transpose(1, 2)
# _, *size = x.shape

# ekm = ft.EKM(
#     size=size,
#     compression=10,
#     alpha=0.1,
#     num_iters=10,
#     num_grad_steps=None,
#     seed=42,
#     verbose=True,
# )
# u, v = ekm.decompose(x)
# print(f"EKM: FLOPS = {ekm.flops}")
# print(f"EKM: #zeros = {(u<1e-8).sum()}")
# loss = u.sum()
# loss.backward()

# # %% SVD
# torch.manual_seed(42)
# size = (8, 8, 8, 8)
# x = torch.randn(4, *size, requires_grad=True)
# x = x.flatten(2, -1)
# _, *size = x.shape

# svd = ft.SVD(size=size, compression=10, verbose=True)
# y = svd(x)
# print(y.shape)
# print(f"SVD: FLOPS = {svd.flops}")
# loss = y.sum()
# loss.backward()


# # %% LRMA
# torch.manual_seed(42)
# size = (8, 8, 8, 8)
# x = torch.randn(4, *size, requires_grad=True)
# x = x.flatten(2, -1)
# _, *size = x.shape

# lrma = ft.LRMA(
#     size=size,
#     compression=10,
#     num_iters=5,
#     init="normal",
#     solver="cd",
#     num_grad_steps=None,
#     verbose=True,
# )
# u, v = lrma.decompose(x)
# print(f"LRMA: FLOPS = {lrma.flops}")
# loss = u.sum()
# loss.backward()


# %% NMF
torch.manual_seed(42)
size = (8, 8, 8, 8)
x = torch.rand(4, *size, requires_grad=True).relu()
x = x.flatten(2, -1)  # .transpose(1, -1)
_, *size = x.shape

nmf = ft.NMF(
    size=size,
    compression=3,
    num_iters=100,
    init="uniform",
    solver="hals",
    num_grad_steps=None,
    verbose=True,
)
u, v = nmf.decompose(x)
print(f"NMF: FLOPS = {nmf.flops}")
loss = u.sum()
loss.backward()

# %% WNMF
torch.manual_seed(42)
size = (8, 8, 8, 8)
x = torch.rand(1, *size, requires_grad=True)
x = x.flatten(2, -1)  # .transpose(1, -1)
_, *size = x.shape

nmf = ft.NMF(
    size=size,
    compression=10,
    num_iters=5,
    init="uniform",
    solver="wmu",
    num_grad_steps=None,
    verbose=True,
)
u, v = nmf.decompose(x)
print(f"WNMF: FLOPS = {nmf.flops}")
loss = u.sum()
loss.backward()

#%% MLSVD
torch.manual_seed(42)
size = (8, 8, 8)
x = torch.rand(1, *size, requires_grad=True)

mlsvd = ft.MLSVD(size=size, compression=10, no_grad=False, verbose=True)
y = mlsvd(x)
print(y.shape)
print(f"MLSVD: FLOPS = {mlsvd.flops}")
loss = y.sum()
loss.backward()


# %% NCPD (NTF)
torch.manual_seed(42)
size = (8, 8, 8)
x = torch.rand(1, *size, requires_grad=True)

ncpd = ft.NCPD(
    size=size,
    compression=10,
    num_iters=5,
    init="uniform",
    solver="mu",
    verbose=True,
)
y = ncpd(x)
print(y.shape)
print(f"NTF: FLOPS = {ncpd.flops}")
loss = y.sum()
loss.backward()

# %% NTD
torch.manual_seed(42)
size = (8, 8, 8)
x = torch.randn(1, *size, requires_grad=True).relu()

ntd = ft.NTD(
    size=size,
    compression=10,
    num_iters=5,
    init="uniform",
    solver="mu",
    verbose=True,
)
y = ntd(x)
print(y.shape)
print(f"NTD: FLOPS = {ntd.flops}")
loss = y.sum()
loss.backward()

# %% NTTD
torch.manual_seed(42)
size = (8, 8, 8)
x = torch.rand(1, *size, requires_grad=True)

nttd = ft.NTTD(size=size, compression=10, num_iters=5, verbose=True)
y = nttd(x)
print(y.shape)
print(f"NTTD: FLOPS = {nttd.flops}")
loss = y.sum()
loss.backward()

# # %% Joint NMF
# torch.manual_seed(42)
# query_size = (64 ** 2, 64)
# value_size = (128 ** 2, 96)

# query = torch.rand(4, *query_size)
# key = torch.rand(4, value_size[0], query_size[1])
# value = torch.rand(4, *value_size)

# jnmf = ft.JointNMF(query_size, value_size, rank=10, num_iters=10, verbose=True)
# y = jnmf(query, key, value)
# print(y.shape)


# # %% Joint NTF
# torch.manual_seed(42)
# query_size = (10, 10, 10, 16)
# value_size = (8, 8, 20)

# query = torch.rand(2, *query_size)
# key = torch.rand(2, *value_size[:-1], query_size[-1])
# value = torch.rand(2, *value_size)

# jntf = ft.JointNTF(
#     query_size, value_size, compression=20, num_iters=10, verbose=True
# )
# y = jntf(query, key, value)
# print(y.shape)


# %% FactorizerSubblock (local matricization)
torch.manual_seed(42)
input_dim = output_dim = 64
spatial_size = (64, 64, 64)
x = torch.rand(1, input_dim, *spatial_size)

factorizer_subblock = ft.FactorizerSubblock(
    input_dim,
    output_dim,
    spatial_size,
    tensorize=(ft.Matricize, {"head_dim": 8, "patch_size": 8}),
    act=nn.ReLU,
    factorize=(ft.NMF, {"compression": 10, "init": "uniform", "solver": "mu"}),
)
y = factorizer_subblock(x)
print(factorizer_subblock.factorize.flops["decompose"] * (8 * (8 ** 3)))


# %% FactorizerSubblock (local swin matricization)
torch.manual_seed(42)
input_dim = output_dim = 64
spatial_size = (64, 64, 64)
x = torch.rand(1, input_dim, *spatial_size)

factorizer_subblock = ft.FactorizerSubblock(
    input_dim,
    output_dim,
    spatial_size,
    tensorize=(ft.SWMatricize, {"head_dim": 8, "patch_size": 8}),
    act=nn.ReLU,
    factorize=(ft.NMF, {"compression": 10, "init": "uniform", "solver": "mu"}),
)
y = factorizer_subblock(x)
print(factorizer_subblock.factorize.flops["decompose"] * (2 * 8 * (8 ** 3)))


# %% FactorizerSubblock (global matricization)
torch.manual_seed(42)
input_dim = output_dim = 64
spatial_size = (64, 64, 64)
x = torch.rand(1, input_dim, *spatial_size)

factorizer_subblock = ft.FactorizerSubblock(
    input_dim,
    output_dim,
    spatial_size,
    tensorize=(ft.Matricize, {"head_dim": 8, "grid_size": 1}),
    act=nn.ReLU,
    factorize=(
        ft.NMF,
        {"compression": 10, "init": "uniform", "solver": "mu"},
    ),
)
y = factorizer_subblock(x)
print(factorizer_subblock.factorize.flops["decompose"] * (8))


# %% FactorizerSubblock (local tensorization)
torch.manual_seed(42)
input_dim = output_dim = 64
spatial_size = (64, 64, 64)
x = torch.rand(1, input_dim, *spatial_size)

factorizer_subblock = ft.FactorizerSubblock(
    input_dim,
    output_dim,
    spatial_size,
    tensorize=(ft.Tensorize, {"head_dim": 1, "patch_size": 8}),
    act=nn.ReLU,
    factorize=(
        ft.NTF,
        {"compression": 10, "init": "uniform", "solver": "mu"},
    ),
)
y = factorizer_subblock(x)
print(factorizer_subblock.factorize.flops["decompose"] * (64 * (8 ** 3)))


# %% FactorizerSubblock (local swin tensorization)
torch.manual_seed(42)
input_dim = output_dim = 64
spatial_size = (64, 64, 64)
x = torch.rand(1, input_dim, *spatial_size)

factorizer_subblock = ft.FactorizerSubblock(
    input_dim,
    output_dim,
    spatial_size,
    tensorize=(ft.SWTensorize, {"head_dim": 1, "patch_size": 8}),
    act=nn.ReLU,
    factorize=(
        ft.NTF,
        {"compression": 10, "init": "uniform", "solver": "mu"},
    ),
)
y = factorizer_subblock(x)
print(factorizer_subblock.factorize.flops["decompose"] * (2 * 64 * (8 ** 3)))


# %% FactorizerSubblock (global tensorization)
torch.manual_seed(42)
input_dim = output_dim = 64
spatial_size = (64, 64, 64)
x = torch.rand(1, input_dim, *spatial_size)

factorizer_subblock = ft.FactorizerSubblock(
    input_dim,
    output_dim,
    spatial_size,
    tensorize=(ft.Tensorize, {"head_dim": 1, "grid_size": 1}),
    act=nn.ReLU,
    factorize=(
        ft.NTD,
        {"compression": 10, "init": "uniform", "solver": "mu"},
    ),
)
y = factorizer_subblock(x)
print(factorizer_subblock.factorize.flops["decompose"] * (64))


# %% FactorizerBlock
torch.manual_seed(42)
input_dim = output_dim = 64
spatial_size = (64, 64, 64)
x = torch.rand(1, input_dim, *spatial_size)

factorizer_block = ft.FactorizerBlock(
    input_dim,
    output_dim,
    spatial_size,
    ntf=(
        ft.FactorizerSubblock,
        {
            "tensorize": (ft.Tensorize, {"head_dim": 1, "patch_size": 8}),
            "act": nn.ReLU,
            "factorize": (
                ft.NTD,
                {"compression": 10, "init": "uniform", "solver": "mu"},
            ),
        },
    ),
)
y = factorizer_block(x)
print(factorizer_block.blocks["ntf"].fn[1].factorize.flops)

