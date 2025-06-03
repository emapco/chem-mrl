from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from chem_mrl.schemas import LatentAttentionConfig


class PreNorm(torch.nn.Module):
    def __init__(self, dim: int, fn: Callable):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x, **kwargs) -> Tensor:
        x = self.norm(x)
        return self.fn(x, **kwargs)


class GEGLU(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gates)


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * mult * 2), GEGLU(), torch.nn.Linear(dim * mult, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class Attention(torch.nn.Module):
    def __init__(self, hidden_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head**-0.5
        self.heads = heads

        self.W_Q = torch.nn.Linear(hidden_dim, inner_dim, bias=False)
        self.W_KV = torch.nn.Linear(hidden_dim, inner_dim * 2, bias=False)
        self.W_OUT = torch.nn.Linear(inner_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor, context: Tensor | None = None):
        h = self.heads
        q = self.W_Q(x)
        context = context if context is not None else x
        k, v = self.W_KV(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        try:
            with sdpa_kernel(
                backends=[
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ],
                set_priority=True,
            ):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        except RuntimeError:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.W_OUT(out)


class LatentAttentionLayer(nn.Module):
    def __init__(self, config: LatentAttentionConfig):
        super().__init__()
        self.config = config
        self.cross_attend_blocks = torch.nn.ModuleList(
            [
                PreNorm(
                    config.hidden_dim,
                    Attention(
                        config.hidden_dim,
                        heads=config.num_cross_heads,
                        dim_head=config.cross_head_dim,
                    ),
                ),
                PreNorm(
                    config.hidden_dim,
                    FeedForward(config.hidden_dim),
                ),
            ]
        )
        self.register_parameter(
            "latents",
            torch.nn.Parameter(torch.randn(config.num_latents, config.hidden_dim)),
        )

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        hidden_embeddings = features["token_embeddings"]  # (Batch, SeqLen, Hidden)
        ## cross-attention block
        cross_attn, cross_ff = self.cross_attend_blocks
        b, *_ = hidden_embeddings.shape
        dictionary = repeat(self.latents, "n d -> b n d", b=b)
        hidden_embeddings: Tensor = (
            cross_attn(hidden_embeddings, context=dictionary) + hidden_embeddings
        )
        hidden_embeddings: Tensor = cross_ff(hidden_embeddings) + hidden_embeddings  # (B, S, H)
        if self.config.output_normalize:
            # Normalize each token embedding along the hidden dimension
            hidden_embeddings = torch.nn.functional.normalize(hidden_embeddings, p=2, dim=-1)

        features.update({"token_embeddings": hidden_embeddings})
        # Add latent layer to all embedding layers so it contributes to matryoshka loss
        if "all_layer_embeddings" in features:
            features["all_layer_embeddings"] = (  # type: ignore - all_layer_embeddings is a tuple of Tensors
                *features["all_layer_embeddings"],
                hidden_embeddings,
            )
        return features

    def __repr__(self) -> str:
        return f"LatentAttentionLayer({self.get_config_dict()})"

    def get_config_dict(self) -> dict[str, Any]:
        return self.config.asdict()

    def save(self, output_path) -> None:
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path) -> LatentAttentionLayer:
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)

        return LatentAttentionLayer(LatentAttentionConfig(**config))
