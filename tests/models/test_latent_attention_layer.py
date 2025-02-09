from pathlib import Path

import pytest
import torch

from chem_mrl.models.LatentAttentionLayer import (
    GEGLU,
    Attention,
    FeedForward,
    LatentAttentionLayer,
    PreNorm,
)
from chem_mrl.schemas import LatentAttentionConfig


def test_prenorm():
    dim = 16
    fn = torch.nn.Linear(dim, dim)
    module = PreNorm(dim, fn)
    x = torch.randn(2, 10, dim)
    output = module(x)
    assert output.shape == x.shape


def test_geglu():
    module = GEGLU()
    x = torch.randn(2, 10, 16 * 2)  # GEGLU expects input with dim split into 2
    output = module(x)
    assert output.shape == (2, 10, 16)


def test_feedforward():
    dim = 16
    module = FeedForward(dim)
    x = torch.randn(2, 10, dim)
    output = module(x)
    assert output.shape == x.shape


def test_attention():
    hidden_dim = 32
    heads = 4
    attn = Attention(hidden_dim, heads)
    x = torch.randn(2, 10, hidden_dim)
    output = attn(x)
    assert output.shape == x.shape


def test_latent_attention_layer():
    config = LatentAttentionConfig(
        hidden_dim=32, num_cross_heads=4, cross_head_dim=8, num_latents=6, output_normalize=True
    )
    layer = LatentAttentionLayer(config)
    token_embeddings = torch.randn(2, 10, config.hidden_dim)
    features = {"token_embeddings": token_embeddings}
    output = layer(features)
    assert "token_embeddings" in output
    assert output["token_embeddings"].shape == token_embeddings.shape


def test_latent_attention_layer_save_load(tmp_path: Path):
    # test methods required by sentences transformers to properly save and load a module
    config = LatentAttentionConfig(
        hidden_dim=32, num_cross_heads=4, cross_head_dim=8, num_latents=6, output_normalize=True
    )
    layer = LatentAttentionLayer(config)
    layer.save(tmp_path)
    loaded_layer = LatentAttentionLayer.load(tmp_path)
    assert loaded_layer.get_config_dict() == layer.get_config_dict()


if __name__ == "__main__":
    pytest.main()
