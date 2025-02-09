# type: ignore
import pytest

from chem_mrl.schemas import LatentAttentionConfig


def test_latent_attention_config_type_validation():
    with pytest.raises(TypeError):
        LatentAttentionConfig(hidden_dim="not an int")
    with pytest.raises(TypeError):
        LatentAttentionConfig(num_latents="not an int")
    with pytest.raises(TypeError):
        LatentAttentionConfig(num_cross_heads="not an int")
    with pytest.raises(TypeError):
        LatentAttentionConfig(cross_head_dim="not an int")
    with pytest.raises(TypeError):
        LatentAttentionConfig(output_normalize="not a bool")
    with pytest.raises(TypeError):
        LatentAttentionConfig(enable="not a bool")


def test_latent_attention_config_value_validation():
    with pytest.raises(ValueError):
        LatentAttentionConfig(hidden_dim=0)
    with pytest.raises(ValueError):
        LatentAttentionConfig(num_latents=0)
    with pytest.raises(ValueError):
        LatentAttentionConfig(num_cross_heads=0)
    with pytest.raises(ValueError):
        LatentAttentionConfig(cross_head_dim=0)
