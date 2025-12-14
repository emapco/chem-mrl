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


def test_prepare_attention_mask_none() -> None:
    """Test _prepare_attention_mask returns None when input is None."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")
    assert model._prepare_attention_mask(None, 10) is None


def test_prepare_attention_mask_matching_length() -> None:
    """Test _prepare_attention_mask with matching lengths."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")
    batch_size, seq_len = 2, 5
    # 1 for valid, 0 for padding
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float)

    key_padding_mask = model._prepare_attention_mask(attention_mask, seq_len)

    assert key_padding_mask is not None
    assert key_padding_mask.shape == (batch_size, seq_len)
    assert key_padding_mask.dtype == torch.bool

    # Expected: True for padding (where input was 0), False for valid (where input was 1)
    expected = torch.tensor([[False, False, False, True, True], [False, False, False, False, False]], dtype=torch.bool)
    assert torch.equal(key_padding_mask, expected)


def test_prepare_attention_mask_truncation() -> None:
    """Test _prepare_attention_mask truncates when mask is longer than target."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")
    batch_size = 2
    mask_len = 8
    target_len = 5

    attention_mask = torch.ones(batch_size, mask_len)
    # Set some padding in the part that should be truncated
    attention_mask[:, 6:] = 0
    # Set some padding in the kept part
    attention_mask[0, 4] = 0

    key_padding_mask = model._prepare_attention_mask(attention_mask, target_len)

    assert key_padding_mask is not None
    assert key_padding_mask.shape == (batch_size, target_len)

    # The mask should be truncated to the first 5 elements
    # Original first 5: [1, 1, 1, 1, 0] for batch 0
    expected_0 = torch.tensor([False, False, False, False, True], dtype=torch.bool)
    assert torch.equal(key_padding_mask[0], expected_0)


def test_prepare_attention_mask_padding() -> None:
    """Test _prepare_attention_mask pads when mask is shorter than target."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")
    batch_size = 2
    mask_len = 3
    target_len = 5

    attention_mask = torch.ones(batch_size, mask_len)
    # Set a padding in the existing mask
    attention_mask[0, 2] = 0

    key_padding_mask = model._prepare_attention_mask(attention_mask, target_len)

    assert key_padding_mask is not None
    assert key_padding_mask.shape == (batch_size, target_len)

    # The mask should be padded with zeros (padding tokens), then inverted to True (ignore)
    # Original: [1, 1, 0] -> Inverted: [F, F, T]
    # Padded part: [0, 0] -> Inverted: [T, T]
    expected_0 = torch.tensor([False, False, True, True, True], dtype=torch.bool)
    assert torch.equal(key_padding_mask[0], expected_0)


def test_prepare_attention_mask_shape_and_dtype() -> None:
    """Test _prepare_attention_mask handles 3D input and returns correct dtype."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")
    batch_size, seq_len = 2, 5

    # 3D input (e.g. from some tokenizers or models)
    attention_mask = torch.ones(batch_size, 1, seq_len)

    key_padding_mask = model._prepare_attention_mask(attention_mask, seq_len)

    assert key_padding_mask is not None
    assert key_padding_mask.shape == (batch_size, seq_len)
    assert key_padding_mask.dtype == torch.bool


def test_prepare_attention_mask_1d_input() -> None:
    """Test _prepare_attention_mask handles 1D input by adding batch dimension."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")
    seq_len = 5

    # 1D input (unbatched)
    attention_mask = torch.tensor([1, 1, 1, 0, 0])

    key_padding_mask = model._prepare_attention_mask(attention_mask, seq_len)

    assert key_padding_mask is not None
    assert key_padding_mask.shape == (1, seq_len)  # Should add batch dimension
    assert key_padding_mask.dtype == torch.bool

    expected = torch.tensor([[False, False, False, True, True]], dtype=torch.bool)
    assert torch.equal(key_padding_mask, expected)


def test_prepare_attention_mask_incompatible_4d() -> None:
    """Test _prepare_attention_mask raises error for incompatible 4D tensors."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")

    # 4D input that can't be squeezed to 2D
    attention_mask = torch.ones(2, 3, 4, 5)

    with pytest.raises(ValueError, match="incompatible shape"):
        model._prepare_attention_mask(attention_mask, 5)


def test_prepare_attention_mask_all_padding() -> None:
    """Test _prepare_attention_mask handles sequences that are all padding."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")
    batch_size, seq_len = 2, 5

    # All padding (all zeros)
    attention_mask = torch.zeros(batch_size, seq_len)

    key_padding_mask = model._prepare_attention_mask(attention_mask, seq_len)

    assert key_padding_mask is not None
    assert key_padding_mask.shape == (batch_size, seq_len)

    # All positions should be True (ignored)
    assert torch.all(key_padding_mask).item()


def test_prepare_attention_mask_no_padding() -> None:
    """Test _prepare_attention_mask handles sequences with no padding."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")
    batch_size, seq_len = 2, 5

    # No padding (all ones)
    attention_mask = torch.ones(batch_size, seq_len)

    key_padding_mask = model._prepare_attention_mask(attention_mask, seq_len)

    assert key_padding_mask is not None
    assert key_padding_mask.shape == (batch_size, seq_len)

    # All positions should be False (attend)
    assert not torch.any(key_padding_mask).item()


def test_prepare_attention_mask_huggingface_convention() -> None:
    """Test _prepare_attention_mask correctly inverts HuggingFace convention.

    HuggingFace: 1 = valid token, 0 = padding
    PyTorch key_padding_mask: True = ignore, False = attend
    """
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")

    # Create a mask following HuggingFace convention
    # Batch 0: 3 tokens + 2 padding
    # Batch 1: 5 tokens + 0 padding
    # Batch 2: 1 token + 4 padding
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
        ],
        dtype=torch.float,
    )

    key_padding_mask = model._prepare_attention_mask(attention_mask, 5)

    # Verify inversion: HF '1' -> PT 'False', HF '0' -> PT 'True'
    expected = torch.tensor(
        [
            [False, False, False, True, True],  # 3 attend, 2 ignore
            [False, False, False, False, False],  # 5 attend, 0 ignore
            [False, True, True, True, True],  # 1 attend, 4 ignore
        ],
        dtype=torch.bool,
    )

    assert torch.equal(key_padding_mask, expected)


def test_prepare_attention_mask_batch_variation() -> None:
    """Test _prepare_attention_mask handles batches with varying sequence lengths."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")

    # Simulate different actual sequence lengths in a batch
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],  # 8 tokens (no padding)
            [1, 1, 1, 1, 1, 0, 0, 0],  # 5 tokens (3 padding)
            [1, 1, 0, 0, 0, 0, 0, 0],  # 2 tokens (6 padding)
        ],
        dtype=torch.float,
    )

    key_padding_mask = model._prepare_attention_mask(attention_mask, 8)

    # Count ignored positions per batch
    num_ignored = key_padding_mask.sum(dim=1)
    expected_ignored = torch.tensor([0, 3, 6])

    assert torch.equal(num_ignored, expected_ignored)


def test_prepare_attention_mask_int_dtype() -> None:
    """Test _prepare_attention_mask handles integer dtype attention masks."""
    model = MaxPoolBERT(word_embedding_dimension=768, pooling_strategy="mha")

    # Integer dtype (common from tokenizers)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    key_padding_mask = model._prepare_attention_mask(attention_mask, 5)

    assert key_padding_mask is not None
    assert key_padding_mask.dtype == torch.bool

    expected = torch.tensor([[False, False, False, True, True]], dtype=torch.bool)
    assert torch.equal(key_padding_mask, expected)
