# type: ignore
# Copyright 2025 Emmanuel Cortes. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from chem_mrl.schemas.Enums import MaxPoolBERTStrategyOption
from chem_mrl.schemas.MaxPoolBERTConfig import MaxPoolBERTConfig


def test_maxpoolbert_config_defaults():
    """Test default values for MaxPoolBERTConfig."""
    config = MaxPoolBERTConfig()
    assert config.enable is False
    assert config.num_attention_heads == 4
    assert config.last_k_layers == 3
    assert config.pooling_strategy == MaxPoolBERTStrategyOption.mha


def test_maxpoolbert_config_custom_values():
    """Test custom values for MaxPoolBERTConfig."""
    config = MaxPoolBERTConfig(
        enable=True,
        num_attention_heads=8,
        last_k_layers=6,
        pooling_strategy=MaxPoolBERTStrategyOption.mean_seq_mha,
    )
    assert config.enable is True
    assert config.num_attention_heads == 8
    assert config.last_k_layers == 6
    assert config.pooling_strategy == MaxPoolBERTStrategyOption.mean_seq_mha


def test_maxpoolbert_config_type_validation():
    """Test type validation for MaxPoolBERTConfig parameters."""
    with pytest.raises(TypeError, match="enable must be a bool"):
        MaxPoolBERTConfig(enable="true")
    with pytest.raises(TypeError, match="num_attention_heads must be an int"):
        MaxPoolBERTConfig(num_attention_heads=4.0)
    with pytest.raises(TypeError, match="last_k_layers must be an int"):
        MaxPoolBERTConfig(last_k_layers=3.5)


def test_maxpoolbert_config_value_validation():
    """Test value validation for MaxPoolBERTConfig parameters."""
    with pytest.raises(ValueError, match="pooling_strategy must be one of"):
        MaxPoolBERTConfig(pooling_strategy="invalid_strategy")
    with pytest.raises(ValueError, match="pooling_strategy must be one of"):
        MaxPoolBERTConfig(pooling_strategy=123)
    with pytest.raises(ValueError, match="num_attention_heads must be positive"):
        MaxPoolBERTConfig(num_attention_heads=0)
    with pytest.raises(ValueError, match="num_attention_heads must be positive"):
        MaxPoolBERTConfig(num_attention_heads=-1)
    with pytest.raises(ValueError, match="last_k_layers must be positive"):
        MaxPoolBERTConfig(last_k_layers=0)
    with pytest.raises(ValueError, match="last_k_layers must be positive"):
        MaxPoolBERTConfig(last_k_layers=-2)


@pytest.mark.parametrize("strategy", MaxPoolBERTStrategyOption)
def test_maxpoolbert_config_all_strategies(strategy):
    """Test that all pooling strategies are valid."""
    config = MaxPoolBERTConfig(pooling_strategy=strategy)
    assert config.pooling_strategy == strategy


def test_maxpoolbert_config_asdict():
    """Test asdict method."""
    config = MaxPoolBERTConfig(
        enable=True,
        num_attention_heads=8,
        last_k_layers=5,
        pooling_strategy=MaxPoolBERTStrategyOption.mha,
    )
    config_dict = config.asdict()
    assert isinstance(config_dict, dict)
    assert config_dict["enable"] is True
    assert config_dict["num_attention_heads"] == 8
    assert config_dict["last_k_layers"] == 5
    assert config_dict["pooling_strategy"] == "mha"


def test_maxpoolbert_config_equality():
    """Test equality comparison."""
    config1 = MaxPoolBERTConfig()
    config2 = MaxPoolBERTConfig()
    config3 = MaxPoolBERTConfig(enable=True)

    assert config1 == config2
    assert config1 != config3
    assert config1 != "not_a_config"


def test_maxpoolbert_config_edge_cases():
    """Test edge cases for MaxPoolBERTConfig."""
    # Very large values
    config = MaxPoolBERTConfig(
        num_attention_heads=128,
        last_k_layers=100,
    )
    assert config.num_attention_heads == 128
    assert config.last_k_layers == 100

    # Minimum valid values
    config = MaxPoolBERTConfig(
        num_attention_heads=1,
        last_k_layers=1,
    )
    assert config.num_attention_heads == 1
    assert config.last_k_layers == 1
