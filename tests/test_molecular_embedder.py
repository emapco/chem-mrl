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

import numpy as np
import pytest
import torch

from chem_mrl import ChemMRL

DEFAULT_MODEL_KWARGS = {"attn_implementation": "eager"}


@pytest.fixture
def base_model():
    """Fixture for basic ChemMRL model on CPU."""
    return ChemMRL(device="cpu", model_kwargs=DEFAULT_MODEL_KWARGS)


@pytest.fixture
def half_precision_model():
    """Fixture for half precision model."""
    return ChemMRL(device="cpu", model_kwargs=DEFAULT_MODEL_KWARGS, use_half_precision=True)


@pytest.fixture
def gpu_model():
    """Fixture for GPU model (skipped if CUDA not available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return ChemMRL(device="cuda")


class TestChemMRLInitialization:
    def test_default_initialization(self, base_model):
        assert base_model.device.type == "cpu"
        assert base_model.use_half_precision is False

    def test_with_trust_remote_code(self):
        model = ChemMRL(device="cpu", trust_remote_code=True)
        assert model.device.type == "cpu"

    def test_without_trust_remote_code(self):
        """Should raise an exception since the model requires remote code."""
        with pytest.raises(ValueError):
            ChemMRL(device="cpu", trust_remote_code=False)

    @pytest.mark.parametrize("similarity_fn", ["cosine", "tanimoto"])
    def test_similarity_function_initialization(self, similarity_fn):
        model = ChemMRL(device="cpu", model_kwargs=DEFAULT_MODEL_KWARGS, similarity_fn_name=similarity_fn)
        assert model.similarity_fn_name == similarity_fn

    def test_backbone_property_returns_self(self, base_model):
        assert base_model.backbone is base_model
        assert hasattr(base_model, "encode")
        assert hasattr(base_model, "tokenize")


class TestChemMRLHalfPrecision:
    def test_disabled_by_default(self, base_model):
        assert base_model.use_half_precision is False

    def test_enabled(self, half_precision_model):
        assert half_precision_model.use_half_precision is True

    @pytest.mark.parametrize("convert_to_tensor,convert_to_numpy", [(True, False), (False, True)])
    def test_output_dtype(self, half_precision_model, convert_to_tensor, convert_to_numpy):
        embeddings = half_precision_model.embed(
            "CCO", convert_to_tensor=convert_to_tensor, convert_to_numpy=convert_to_numpy
        )
        if convert_to_tensor:
            assert embeddings.dtype == torch.float16
        else:
            assert embeddings.dtype == np.float16


class TestChemMRLEmbed:
    @pytest.mark.parametrize(
        "smiles,expected_shape",
        [
            ("CCO", 1),  # Single SMILES -> 1D array
            (["CCO", "CC(C)O", "c1ccccc1"], 2),  # Multiple SMILES -> 2D array
            (np.array(["CCO", "CC(C)O", "c1ccccc1"]), 2),  # Numpy input
            ([], 1),  # Empty list
        ],
    )
    def test_embed_input_shapes(self, base_model, smiles, expected_shape):
        embeddings = base_model.embed(smiles)
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings.shape) == expected_shape
        if isinstance(smiles, list) and smiles:
            assert embeddings.shape[0] == len(smiles)

    def test_embed_with_batch_size(self, base_model):
        smiles_list = ["CCO", "CC(C)O", "c1ccccc1", "CC(=O)O"]
        embeddings = base_model.embed(smiles_list, batch_size=2)
        assert embeddings.shape[0] == 4

    def test_embed_output_tensor(self, base_model):
        embeddings = base_model.embed("CCO", convert_to_tensor=True, convert_to_numpy=False)
        assert isinstance(embeddings, torch.Tensor)

    def test_normalize_embeddings(self, base_model):
        embeddings = base_model.embed(["CCO", "CC(C)O"], convert_to_tensor=True, normalize_embeddings=True)
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_precision_float32(self, base_model):
        embeddings = base_model.embed("CCO", precision="float32", convert_to_tensor=True)
        assert embeddings.dtype == torch.float32, "Expected float32 embeddings from base model"

    def test_precision_float16(self, half_precision_model) -> None:
        embeddings = half_precision_model.embed("CCO", precision="float32", convert_to_tensor=True)
        assert embeddings.dtype == torch.float16, "Expected float16 embeddings from half-precision model"


class TestChemMRLSimilarity:
    @pytest.mark.parametrize(
        "similarity_fn,min_val,max_val",
        [
            ("cosine", -1.0, 1.0),
            ("tanimoto", 0.0, 1.0),
        ],
    )
    def test_similarity_functions(self, similarity_fn, min_val, max_val):
        model = ChemMRL(device="cpu", model_kwargs=DEFAULT_MODEL_KWARGS, similarity_fn_name=similarity_fn)
        embeddings1 = model.embed(["CCO", "CC(C)O"], convert_to_tensor=True)
        embeddings2 = model.embed(["c1ccccc1", "CC(=O)O"], convert_to_tensor=True)

        similarities = model.similarity(embeddings1, embeddings2)

        assert similarities.shape == (2, 2)
        assert torch.all(similarities >= min_val)
        assert torch.all(similarities <= max_val)

    def test_similarity_pairwise(self, base_model):
        embeddings = base_model.embed(["CCO", "CC(C)O", "c1ccccc1"], convert_to_tensor=True)
        similarities = base_model.similarity_pairwise(embeddings, embeddings)

        assert similarities.shape == (3,)
        assert torch.allclose(similarities, torch.ones_like(similarities), atol=1e-5)


class TestChemMRLTruncateDim:
    def test_truncate_dim_initialization(self):
        model = ChemMRL(device="cpu", model_kwargs=DEFAULT_MODEL_KWARGS, truncate_dim=128)
        assert model.truncate_dim == 128

    def test_truncate_dim_in_embed(self, base_model):
        full_embeddings = base_model.embed("CCO")
        full_dim = full_embeddings.shape[0]

        truncated_embeddings = base_model.embed("CCO", truncate_dim=64)
        truncated_dim = truncated_embeddings.shape[0]

        assert truncated_dim == 64
        assert truncated_dim < full_dim


class TestChemMRLEdgeCases:
    @pytest.mark.parametrize(
        "smiles",
        [
            "[Fe+2]%@.#-=#@@[Fe++]",  # Special characters
            "C" * 500,  # Very long SMILES
        ],
    )
    def test_embed_special_inputs(self, base_model, smiles):
        embedding = base_model.embed(smiles)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0

    def test_deterministic_embed_calls(self, base_model):
        smiles = "CCO"
        embedding1 = base_model.embed(smiles, convert_to_tensor=True)
        embedding2 = base_model.embed(smiles, convert_to_tensor=True)
        assert torch.allclose(embedding1, embedding2, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestChemMRLGPU:
    def test_gpu_initialization(self, gpu_model):
        assert gpu_model.device.type == "cuda"

    def test_gpu_embedding(self, gpu_model):
        smiles_list = ["CCO", "CC(C)O", "c1ccccc1"]
        embeddings = gpu_model.embed(smiles_list, convert_to_tensor=True)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.device.type == "cuda"
        assert embeddings.shape == (3, embeddings.shape[1])

    def test_gpu_similarity_computation(self):
        model = ChemMRL(device="cuda", similarity_fn_name="tanimoto")
        embeddings1 = model.embed(["CCO", "CC(C)O"], convert_to_tensor=True)
        embeddings2 = model.embed(["c1ccccc1", "CC(=O)O"], convert_to_tensor=True)

        similarities = model.similarity(embeddings1, embeddings2)

        assert similarities.device.type == "cuda"
        assert similarities.shape == (2, 2)
        assert torch.all(similarities >= 0.0) and torch.all(similarities <= 1.0)

    def test_gpu_half_precision(self):
        model = ChemMRL(device="cuda", use_half_precision=True)
        embeddings = model.embed("CCO", convert_to_tensor=True)

        assert embeddings.device.type == "cuda"
        assert embeddings.dtype == torch.float16

    def test_gpu_compile(self):
        compile_config = {"backend": "inductor", "mode": "default", "fullgraph": False, "dynamic": True}
        model = ChemMRL(device="cuda", compile_kwargs=compile_config)
        embeddings = model.embed("CCO", convert_to_tensor=True)

        assert embeddings.device.type == "cuda"

    def test_gpu_cpu_consistency(self, base_model, gpu_model):
        smiles = "CCO"
        embedding_cpu = base_model.embed(smiles, convert_to_tensor=True)
        embedding_gpu = gpu_model.embed(smiles, convert_to_tensor=True).cpu()

        assert torch.allclose(embedding_cpu, embedding_gpu, atol=1e-3, rtol=1e-3)
