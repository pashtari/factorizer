# %%
import torch

from factorizer.layers.attention import (
    MLP,
    MultiheadAttention,
    FastAttention,
    NystromAttention,
    KernelAttention,
    TransformerEncoder,
)

# %% MLP
torch.manual_seed(0)
B = 1
N = 128
E = 32
x = torch.randn(B, N, E)

mlp = MLP(E, 4 * E)
y = mlp(x)
print(y.shape)

# %% Multihead attention
torch.manual_seed(0)
B = 1
N = 128
E = 32
x = torch.randn(B, N, E)

attn = MultiheadAttention(E, num_heads=4)
y = attn(x, x, x)
print(y.shape)
# %timeit attn(x, x, x)

# %% Performer
torch.manual_seed(0)
B = 1
N = 128
E = 32
x = torch.randn(B, N, E)

fast_attn = FastAttention(E, num_heads=4)
y1 = fast_attn(x, x, x)
print(y1.shape)
print(f"error: {(y-y1).abs().mean()}")
# %timeit fast_attn(x, x, x)

# %% Nystrom attention
torch.manual_seed(0)

B = 1
N = 128
E = 32
x = torch.randn(B, N, E)

nystrom_attn = NystromAttention(E, num_heads=4, num_landmarks=8)
y2 = nystrom_attn(x, x, x)
print(y2.shape)
print(f"error: {(y-y2).abs().mean()}")
# %timeit nystrom_attn(x, x, x)

# %% Kernel attention
torch.manual_seed(0)
B = 1
N = 128
E = 32
x = torch.randn(B, N, E)

kernel_attn = KernelAttention(E, num_heads=4)
y3 = kernel_attn(x, x, x)
print(y3.shape)
print(f"error: {(y-y3).abs().mean()}")
# %timeit kernel_attn(x, x, x)


# # %% TransformerEncoder
# torch.manual_seed(0)
# B = 5
# N = 64
# E = 16
# x = torch.randn(B, N, E)

# enc = TransformerEncoder(
#     E,
#     depth=6,
#     mlp_dim=4 * E,
#     attention=MultiheadAttention,
#     attention_params={"num_heads": 4, "dropout": 0.1},
#     mlp=MLP,
#     mlp_params={"dropout": 0.1},
# )
# y = enc(x)[-1]
# print(y.shape)


# %% performer
import torch
from performer_pytorch import SelfAttention

torch.manual_seed(0)
B = 1
N = 128
E = 32
x = torch.randn(B, N, E)

per = SelfAttention(dim=E, heads=4)
y4 = per(x)
print(y4.shape)
print(f"error: {(y-y4).abs().mean()}")
# %timeit per(x)

# %%
import torch
from nystrom_attention import NystromAttention

torch.manual_seed(0)
B = 1
N = 128
E = 32
x = torch.randn(B, N, E)

nys = NystromAttention(
    dim=E,
    dim_head=8,
    heads=4,
    num_landmarks=64,  # number of landmarks
    pinv_iterations=5,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
)

y5 = nys(x)
print(y.shape)
print(f"error: {(y-y5).abs().mean()}")
# %timeit nys(x)

# %%
