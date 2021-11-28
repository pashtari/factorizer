import math

import torch
from torch import nn
from torch.nn.modules.utils import _pair, _triple
import einops

from .utils.helpers import prod, wrap_class


class PosEmbed(nn.Module):
    def __init__(self, input_dim, spatial_size, dropout=0.0, **kwargs):
        super().__init__()
        length = prod(spatial_size)
        self.pos_embed = nn.Parameter(torch.empty(1, length, input_dim))
        nn.init.normal_(self.pos_embed, std=1.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + self.pos_embed
        out = self.dropout(out)
        return out


class PosEmbed1d(nn.Module):
    def __init__(self, input_dim, spatial_size, dropout=0.0, **kwargs):
        super().__init__()

        if input_dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional embedding with "
                "odd dim (got dim={:d})".format(input_dim)
            )

        self.dimension = input_dim
        self.scale = 1 / math.sqrt(self.input_dim)
        self.length = prod(spatial_size)

        position = torch.arange(0, self.length).unsqueeze(1)
        frequency = torch.exp(
            -math.log(10000.0) * torch.arange(0, input_dim, 2) / input_dim
        ).unsqueeze(0)

        pe = torch.zeros(1, self.length, input_dim)
        pe[:, :, 0::2] = torch.sin(position * frequency)
        pe[:, :, 1::2] = torch.cos(position * frequency)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x * self.scale
        out = out + self.pe[:, : out.shape[1], :]
        out = self.dropout(out)
        return out


class PatchEmbed2d(nn.Module):
    def __init__(self, num_channels, embed_dim, patch_size=1, **kwargs):
        super().__init__()
        patch_size = _pair(patch_size)
        self.conv = nn.Conv2d(
            num_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = einops.rearrange(x, "b e h w -> b (h w) e")
        return x


class PatchEmbed3d(nn.Module):
    def __init__(self, num_channels, embed_dim, patch_size=1, **kwargs):
        super().__init__()
        patch_size = _triple(patch_size)
        self.conv = nn.Conv3d(
            num_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = einops.rearrange(x, "b e h w d -> b (h w d) e")
        return x


class PatchFlat2d(nn.Module):
    def __init__(self, patch_size=1, **kwargs):
        super().__init__()
        self.patch_size = _pair(patch_size)

    def forward(self, x):
        p1, p2 = self.patch_size
        x = einops.rearrange(
            x, "b c (g1 p1) (g2 p2) -> b (g1 g2) (c p1 p2)", p1=p1, p2=p2
        )
        return x


class PatchFlat3d(nn.Module):
    def __init__(self, patch_size, **kwargs):
        super().__init__()
        self.patch_size = _triple(patch_size)

    def forward(self, x):
        p1, p2, p3 = self.patch_size
        x = einops.rearrange(
            x,
            "b c (g1 p1) (g2 p2) (g3 p3) -> b (g1 g2 g3) (c p1 p2 p3)",
            p1=p1,
            p2=p2,
            p3=p3,
        )
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=None,
        ratio=1,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        hidden_dim = ratio * input_dim if hidden_dim is None else hidden_dim

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        hidden_dim=None,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        output_dim = input_dim if output_dim is None else output_dim
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.block(x)


class MultiheadAttention(nn.Module):
    """Multihead attention layer."""

    def __init__(
        self,
        input_dim,
        output_dim=None,
        num_heads=None,
        head_dim=None,
        dropout=0.0,
        bias=False,
        key_dim=None,
        value_dim=None,
        **kwargs,
    ):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        key_dim = input_dim if key_dim is None else key_dim
        value_dim = output_dim if value_dim is None else value_dim

        if num_heads is not None:
            self.num_heads = num_heads
        elif head_dim is not None:
            self.num_heads = key_dim // head_dim
        else:
            self.head_dim = 1

        self.head_dim = key_dim // self.num_heads

        self.q_proj = nn.Linear(input_dim, key_dim, bias=bias)
        self.k_proj = nn.Linear(input_dim, key_dim, bias=bias)
        self.v_proj = nn.Linear(input_dim, value_dim, bias=bias)

        self.out_proj = nn.Linear(value_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, mask=None):
        scale = 1 / math.sqrt(key.shape[-1])
        attn = torch.einsum("b h n e, b h m e -> b h n m", query, key) * scale
        if mask is not None:
            attn = attn.masked_fill(mask[None], -math.inf)

        attn = attn.softmax(dim=-1)
        attn = torch.einsum("b h n m, b h m e -> b h n e", attn, value)
        return attn

    def forward(self, query, key=None, value=None, mask=None, **kwargs):
        # query: B × M × C, key: B × N × C, value: B × N × D
        key = query if key is None else key
        value = key if value is None else value

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # fold the embedding dim to make heads
        h = self.num_heads
        e = self.head_dim
        query = einops.rearrange(query, "b n (h e) -> b h n e", h=h, e=e)
        key = einops.rearrange(key, "b n (h e) -> b h n e", h=h, e=e)
        value = einops.rearrange(value, "b n (h e) -> b h n e", h=h, e=e)

        # get attention
        out = self.attention(query, key, value, mask)

        # flatten the embedding and head dim
        out = einops.rearrange(out, "b h n e -> b n (h e)")

        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class FastAttention(nn.Module):
    """Fast attention layer based on Performer."""

    def __init__(
        self,
        input_dim,
        output_dim=None,
        num_heads=None,
        head_dim=None,
        dropout=0.0,
        bias=False,
        key_dim=None,
        value_dim=None,
        kernel_fun=nn.ReLU(),
        **kwargs,
    ):
        super().__init__()
        output_dim = input_dim if output_dim is None else output_dim
        key_dim = input_dim if key_dim is None else key_dim
        value_dim = output_dim if value_dim is None else value_dim

        if num_heads is not None:
            self.num_heads = num_heads
        elif head_dim is not None:
            self.num_heads = key_dim // head_dim
        else:
            self.head_dim = 1

        self.head_dim = key_dim // self.num_heads

        self.scale = 1 / math.sqrt(self.head_dim)
        self.kernel_fun = kernel_fun

        self.q_proj = nn.Linear(input_dim, key_dim, bias=bias)
        self.k_proj = nn.Linear(input_dim, key_dim, bias=bias)
        self.v_proj = nn.Linear(input_dim, value_dim, bias=bias)

        orth_proj = []
        for _ in range(self.num_heads):
            proj = torch.randn(self.head_dim, self.head_dim)
            proj, _ = torch.qr(proj)
            orth_proj.append(proj)

        orth_proj = torch.stack(orth_proj)
        self.register_buffer("orth_proj", orth_proj)

        self.out_proj = nn.Linear(value_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def get_attention(self, query, key, value):
        phi_q = self.kernel_fun(
            torch.einsum("b h n e, h e d -> b h n d", query, self.orth_proj)
        )
        phi_k = self.kernel_fun(
            torch.einsum("b h n e, h e d -> b h n d", key, self.orth_proj)
        )

        # normalize
        phi_k_rowsum = phi_k.sum(-2)
        scale = torch.einsum("... n d, ... d -> ... n", phi_q, phi_k_rowsum)
        scale = 1 / (self.scale * scale + 1e-8)
        phi_q = torch.einsum("... n d, ... n -> ... n d", phi_q, scale)

        kv = torch.einsum("... n d, ... n e -> ... d e", phi_k, value)
        attn = torch.einsum("... n d, ... d e -> ... n e", phi_q, kv)
        return attn

    def forward(self, query, key=None, value=None, **kwargs):
        # query: B × M × C, key: B × N × C, value: B × N × D
        key = query if key is None else key
        value = key if value is None else value

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # fold the embedding dim to make heads
        h = self.num_heads
        e = self.head_dim
        query = einops.rearrange(query, "b n (h e) -> b h n e", h=h, e=e)
        key = einops.rearrange(key, "b n (h e) -> b h n e", h=h, e=e)
        value = einops.rearrange(value, "b n (h e) -> b h n e", h=h, e=e)

        # get attention
        out = self.get_attention(query, key, value)

        # flatten the embedding and head dim
        out = einops.rearrange(out, "b h n e -> b n (h e)")

        out = self.out_proj(out)
        out = self.dropout(out)
        return out


class TransformerEncoderBlock(nn.Module):
    "Transformer encoder block. "

    def __init__(self, input_dim, attention, mlp, **kwargs):
        super().__init__()
        if attention:
            attention = wrap_class(attention)
            self.norm1 = nn.LayerNorm(input_dim)
            self.self_attn = attention(input_dim)

        if mlp:
            mlp = wrap_class(mlp)
            self.norm2 = nn.LayerNorm(input_dim)
            self.mlp = mlp(input_dim)

    def forward(self, x, *args, **kwargs):
        if hasattr(self, "self_attn"):
            out1 = self.norm1(x)
            out1 = self.self_attn(out1, out1, out1, *args, **kwargs)
            out1 = out1 + x
        else:
            out1 = x

        if hasattr(self, "mlp"):
            out2 = self.norm2(out1)
            out2 = self.mlp(out2)
            out2 = out2 + out1
        else:
            out2 = out1

        return out2


class TransformerEncoder(nn.Module):
    "Transformer encoder for sequence-to-sequence modeling."

    def __init__(
        self,
        input_dim,
        depth,
        attention=MultiheadAttention,
        mlp=MLP,
        **kwargs,
    ):

        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(
                TransformerEncoderBlock(input_dim, attention, mlp)
            )

    def forward(self, x, *args, **kwargs):
        out = x
        for block in self.blocks:
            out = block(out, *args, **kwargs)

        return out


class TransformerDecoderBlock(nn.Module):
    "Transformer decoder block."

    def __init__(self, input_dim, attention, mlp, **kwargs):
        super().__init__()
        if attention:
            # self-attention
            attention = wrap_class(attention)
            self.norm1 = nn.LayerNorm(input_dim)
            self.self_attn = attention(input_dim)

            # encoder-decoder attention
            attention = wrap_class(attention)
            self.norm2 = nn.LayerNorm(input_dim)
            self.attn = attention(input_dim)

        # MLP
        if mlp:
            mlp = wrap_class(mlp)
            self.norm3 = nn.LayerNorm(input_dim)
            self.mlp = mlp(input_dim)

    def forward(self, x1, x2, **kwargs):
        if hasattr(self, "self_attn"):
            out1 = self.norm1(x1)
            out1 = self.self_attn(out1, out1, out1, **kwargs)
            out1 = out1 + x1
        else:
            out1 = x1

        if hasattr(self, "attn"):
            out2 = self.norm2(out1)
            x2 = self.norm2(x2)
            out2 = self.attn(out2, x2, x2, **kwargs)
            out2 = out2 + out1
        else:
            out2 = out1

        if hasattr(self, "mlp"):
            out3 = self.norm3(out2)
            out3 = self.mlp(out3)
            out3 = out3 + out2
        else:
            out3 = out2

        return out3


class TransformerDecoder(nn.Module):
    "Transformer decoder for sequence-to-sequence modeling."

    def __init__(
        self,
        input_dim,
        depth,
        attention=MultiheadAttention,
        mlp=MLP,
        **kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(
                TransformerDecoderBlock(input_dim, attention, mlp)
            )

    def forward(self, x1, x2, **kwargs):
        out = x1
        for block in self.blocks:
            out = block(out, x2, **kwargs)

        return out


class VisionTransformer(nn.Module):
    "Vision transformer for image classification."

    def __init__(
        self,
        num_channels,
        num_classes,
        embed_size,
        embed_dim=512,
        depth=6,
        attention=MultiheadAttention,
        mlp=MLP,
        stem=PatchEmbed2d,
        head=nn.Linear,
        pos_embed=PosEmbed,
        **kwargs,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.embed_size = embed_size
        self.embed_dim = embed_dim
        self.depth = depth

        head = wrap_class(head)
        stem = wrap_class(stem)
        pos_embed = wrap_class(pos_embed)

        self.stem = stem(num_channels, embed_dim)
        self.pos_embed = pos_embed(embed_dim, embed_size)

        self.encoder = TransformerEncoder(embed_dim, depth, attention, mlp)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), head(embed_dim, num_classes)
        )

    def forward(self, x):
        out = self.stem(x)
        out = self.pos_embed(out)
        out = self.encoder(out)
        out = out.mean(1)  # 'b n d -> b d' (global average pooling)
        out = self.head(out)
        return out

