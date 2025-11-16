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

from pathlib import Path

import pytest
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer

from chem_mrl.constants import BASE_MODEL_NAME
from chem_mrl.models import MaxPoolBERT


@pytest.fixture
def transformer_with_hidden_states() -> Transformer:
    """Create a transformer model with output_hidden_states enabled."""
    transformer = Transformer(
        BASE_MODEL_NAME,
        model_args={"trust_remote_code": True},
        config_args={"trust_remote_code": True},
    )
    transformer.auto_model.config.output_hidden_states = True
    return transformer


def test_maxpoolbert_initialization() -> None:
    """Test basic initialization of MaxPoolBERT with different configurations."""
    # Test default initialization
    model = MaxPoolBERT(word_embedding_dimension=768)
    assert model.word_embedding_dimension == 768
    assert model.num_attention_heads == 4
    assert model.last_k_layers == 3
    assert model.pooling_strategy == "mha"
    assert model.multi_head_attention is not None

    # Test custom initialization
    model = MaxPoolBERT(
        word_embedding_dimension=512,
        num_attention_heads=8,
        last_k_layers=6,
        pooling_strategy="max_cls",
    )
    assert model.word_embedding_dimension == 512
    assert model.num_attention_heads == 8
    assert model.last_k_layers == 6
    assert model.pooling_strategy == "max_cls"
    assert model.multi_head_attention is None  # max_cls strategy doesn't use attention


def test_maxpoolbert_invalid_pooling_strategy() -> None:
    """Test that invalid pooling strategy raises ValueError."""
    with pytest.raises(ValueError, match="Invalid pooling_strategy"):
        MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="invalid_strategy")


def test_maxpoolbert_invalid_attention_heads() -> None:
    """Test that invalid num_attention_heads raises ValueError."""
    with pytest.raises(ValueError, match="must be divisible by num_attention_heads"):
        MaxPoolBERT(word_embedding_dimension=768, num_attention_heads=7, pooling_strategy="max_seq_mha")


def test_maxpoolbert_missing_all_layer_embeddings() -> None:
    """Test that missing all_layer_embeddings raises ValueError."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="max_cls")
    features = {"attention_mask": torch.ones(2, 10)}

    with pytest.raises(ValueError, match="MaxPoolBERT requires 'all_layer_embeddings'"):
        model(features)


def test_maxpoolbert_insufficient_layers() -> None:
    """Test that insufficient layers raises ValueError."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="max_cls", last_k_layers=10)

    batch_size, seq_len, hidden_dim = 2, 10, 768
    all_layers = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(6)]
    features = {
        "all_layer_embeddings": all_layers,
        "attention_mask": torch.ones(batch_size, seq_len),
    }

    with pytest.raises(ValueError, match="Not enough layers"):
        model(features)


def test_maxpoolbert_with_attention_mask() -> None:
    """Test MaxPoolBERT handles attention masks correctly."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha", last_k_layers=4)

    batch_size, seq_len, hidden_dim = 2, 10, 768
    all_layers = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(12)]
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 5:] = 0  # Mask out last 5 tokens in first sequence
    attention_mask[1, 8:] = 0  # Mask out last 2 tokens in second sequence

    features = {
        "all_layer_embeddings": all_layers,
        "attention_mask": attention_mask,
    }

    output = model(features)
    assert "sentence_embedding" in output
    assert output["sentence_embedding"].shape == (batch_size, hidden_dim)


def test_maxpoolbert_without_attention_mask() -> None:
    """Test MaxPoolBERT works without attention mask."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha", last_k_layers=4)

    batch_size, seq_len, hidden_dim = 2, 10, 768
    all_layers = [torch.randn(batch_size, seq_len, hidden_dim) for _ in range(12)]
    features = {"all_layer_embeddings": all_layers}

    output = model(features)
    assert "sentence_embedding" in output
    assert output["sentence_embedding"].shape == (batch_size, hidden_dim)


def test_maxpoolbert_save_and_load(tmp_path: Path) -> None:
    """Test saving and loading MaxPoolBERT model."""
    model = MaxPoolBERT(
        word_embedding_dimension=768,
        num_attention_heads=8,
        last_k_layers=5,
        pooling_strategy="max_seq_mha",
    )

    save_dir = tmp_path / "maxpoolbert"
    save_dir.mkdir()
    model.save(str(save_dir))

    loaded_model = MaxPoolBERT.load(str(save_dir))
    assert loaded_model.word_embedding_dimension == 768
    assert loaded_model.num_attention_heads == 8
    assert loaded_model.last_k_layers == 5
    assert loaded_model.pooling_strategy == "max_seq_mha"
    assert loaded_model.multi_head_attention is not None


def test_maxpoolbert_save_and_load_cls_strategy(tmp_path: Path) -> None:
    """Test saving and loading MaxPoolBERT model with cls strategy."""
    model = MaxPoolBERT(
        word_embedding_dimension=768,
        pooling_strategy="max_cls",
        last_k_layers=3,
    )

    save_dir = tmp_path / "maxpoolbert_cls"
    save_dir.mkdir()
    model.save(str(save_dir))

    loaded_model = MaxPoolBERT.load(str(save_dir))
    assert loaded_model.word_embedding_dimension == 768
    assert loaded_model.pooling_strategy == "max_cls"
    assert loaded_model.last_k_layers == 3
    assert loaded_model.multi_head_attention is None


def test_maxpoolbert_different_strategies_produce_different_embeddings(
    transformer_with_hidden_states: Transformer,
) -> None:
    """Test that different pooling strategies produce different embeddings."""
    strategies = ["cls", "max_cls", "mha", "max_seq_mha", "mean_seq_mha", "sum_seq_mha"]
    embeddings_by_strategy = {}

    for strategy in strategies:
        maxpoolbert = MaxPoolBERT(
            word_embedding_dimension=transformer_with_hidden_states.get_word_embedding_dimension(),
            pooling_strategy=strategy,
            last_k_layers=4,
        )
        model = SentenceTransformer(modules=[transformer_with_hidden_states, maxpoolbert])
        embeddings = model.encode(["CCO"], convert_to_tensor=True)
        embeddings_by_strategy[strategy] = embeddings

    # Check that different strategies produce different embeddings
    strategies_list = list(embeddings_by_strategy.keys())
    for i in range(len(strategies_list)):
        for j in range(i + 1, len(strategies_list)):
            emb_i = embeddings_by_strategy[strategies_list[i]]
            emb_j = embeddings_by_strategy[strategies_list[j]]
            # Relax tolerance and include difference in assertion message
            atol = 1e-2
            diff = torch.abs(emb_i - emb_j).max().item()
            assert not torch.allclose(emb_i, emb_j, atol=atol), (
                f"{strategies_list[i]} and {strategies_list[j]} produced nearly identical embeddings "
                f"(max abs diff: {diff:.6f}, atol: {atol})"
            )
