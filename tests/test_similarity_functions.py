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

from chem_mrl.similarity_functions import (
    paired_tanimoto_similarity,
    pairwise_tanimoto_similarity,
    tanimoto_similarity,
)


class TestTanimotoSimilarity:
    """Tests for the full matrix tanimoto_similarity function."""

    def test_tanimoto_similarity_identical_vectors(self):
        """Test that identical vectors have Tanimoto similarity of 1.0."""
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.tensor([[1.0, 2.0, 3.0]])
        result = tanimoto_similarity(a, b)

        assert result.shape == (1, 1)
        assert torch.isclose(result[0, 0], torch.tensor(1.0), atol=1e-6)

    def test_tanimoto_similarity_orthogonal_vectors(self):
        """Test that orthogonal vectors have Tanimoto similarity of 0.0."""
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        result = tanimoto_similarity(a, b)

        assert result.shape == (1, 1)
        assert torch.isclose(result[0, 0], torch.tensor(0.0), atol=1e-6)

    def test_tanimoto_similarity_matrix_shape(self):
        """Test that output has correct shape (n, m) for inputs of shape (n, d) and (m, d)."""
        a = torch.randn(3, 5)
        b = torch.randn(4, 5)
        result = tanimoto_similarity(a, b)

        assert result.shape == (3, 4)

    def test_tanimoto_similarity_range(self):
        """Test that Tanimoto similarity is in range [0, 1] for positive vectors."""
        a = torch.rand(5, 10)
        b = torch.rand(3, 10)
        result = tanimoto_similarity(a, b)

        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_tanimoto_similarity_symmetry(self):
        """Test that T(a, b) == T(b, a).T."""
        a = torch.randn(3, 5)
        b = torch.randn(4, 5)
        result_ab = tanimoto_similarity(a, b)
        result_ba = tanimoto_similarity(b, a)

        assert torch.allclose(result_ab, result_ba.T, atol=1e-6)

    def test_tanimoto_similarity_manual_calculation(self):
        """Test against manual calculation of Tanimoto similarity."""
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = torch.tensor([[2.0, 3.0, 4.0]])

        # Manual calculation: T(a,b) = <a,b> / (||a||^2 + ||b||^2 - <a,b>)
        dot_product = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0  # 20.0
        a_norm_sq = 1.0**2 + 2.0**2 + 3.0**2  # 14.0
        b_norm_sq = 2.0**2 + 3.0**2 + 4.0**2  # 29.0
        expected = dot_product / (a_norm_sq + b_norm_sq - dot_product)  # 20.0 / 23.0

        result = tanimoto_similarity(a, b)
        assert torch.isclose(result[0, 0], torch.tensor(expected), atol=1e-6)

    def test_tanimoto_similarity_with_numpy(self):
        """Test that function works with numpy arrays."""
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.array([[2.0, 3.0, 4.0]])
        result = tanimoto_similarity(a, b)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 1)

    def test_tanimoto_similarity_with_list(self):
        """Test that function works with Python lists."""
        a = [[1.0, 2.0, 3.0]]
        b = [[2.0, 3.0, 4.0]]
        result = tanimoto_similarity(a, b)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 1)

    def test_tanimoto_similarity_batch_computation(self):
        """Test that batch computation is correct."""
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

        result = tanimoto_similarity(a, b)

        # a[0] vs b[0]: identical -> 1.0
        assert torch.isclose(result[0, 0], torch.tensor(1.0), atol=1e-6)
        # a[0] vs b[1]: orthogonal -> 0.0
        assert torch.isclose(result[0, 1], torch.tensor(0.0), atol=1e-6)
        # a[1] vs b[0]: orthogonal -> 0.0
        assert torch.isclose(result[1, 0], torch.tensor(0.0), atol=1e-6)
        # a[1] vs b[1]: identical -> 1.0
        assert torch.isclose(result[1, 1], torch.tensor(1.0), atol=1e-6)

    def test_tanimoto_similarity_zero_denominator_handling(self):
        """Test that zero vectors don't cause division by zero."""
        a = torch.tensor([[0.0, 0.0, 0.0]])
        b = torch.tensor([[1.0, 2.0, 3.0]])
        result = tanimoto_similarity(a, b)

        # Should not raise an error and should return a valid value
        assert not torch.isnan(result[0, 0])
        assert not torch.isinf(result[0, 0])


class TestPairwiseTanimotoSimilarity:
    """Tests for the pairwise (diagonal) pairwise_tanimoto_similarity function."""

    def test_pairwise_tanimoto_identical_vectors(self):
        """Test that identical vectors have pairwise Tanimoto similarity of 1.0."""
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = pairwise_tanimoto_similarity(a, b)

        assert result.shape == (2,)
        assert torch.allclose(result, torch.tensor([1.0, 1.0]), atol=1e-6)

    def test_pairwise_tanimoto_orthogonal_vectors(self):
        """Test that orthogonal vectors have pairwise Tanimoto similarity of 0.0."""
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        result = pairwise_tanimoto_similarity(a, b)

        assert result.shape == (2,)
        assert torch.allclose(result, torch.tensor([0.0, 0.0]), atol=1e-6)

    def test_pairwise_tanimoto_output_shape(self):
        """Test that output has correct shape (n,) for inputs of shape (n, d)."""
        a = torch.randn(5, 10)
        b = torch.randn(5, 10)
        result = pairwise_tanimoto_similarity(a, b)

        assert result.shape == (5,)

    def test_pairwise_tanimoto_range(self):
        """Test that pairwise Tanimoto similarity is in range [0, 1] for positive vectors."""
        a = torch.rand(10, 20)
        b = torch.rand(10, 20)
        result = pairwise_tanimoto_similarity(a, b)

        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

    def test_pairwise_tanimoto_manual_calculation(self):
        """Test against manual calculation of pairwise Tanimoto similarity."""
        a = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        b = torch.tensor([[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]])

        # Manual calculation for first pair
        dot_product_0 = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0  # 20.0
        a_norm_sq_0 = 1.0**2 + 2.0**2 + 3.0**2  # 14.0
        b_norm_sq_0 = 2.0**2 + 3.0**2 + 4.0**2  # 29.0
        expected_0 = dot_product_0 / (a_norm_sq_0 + b_norm_sq_0 - dot_product_0)

        # Manual calculation for second pair
        dot_product_1 = 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0  # 9.0
        a_norm_sq_1 = 2.0**2 + 3.0**2 + 4.0**2  # 29.0
        b_norm_sq_1 = 1.0**2 + 1.0**2 + 1.0**2  # 3.0
        expected_1 = dot_product_1 / (a_norm_sq_1 + b_norm_sq_1 - dot_product_1)

        result = pairwise_tanimoto_similarity(a, b)
        assert torch.isclose(result[0], torch.tensor(expected_0), atol=1e-6)
        assert torch.isclose(result[1], torch.tensor(expected_1), atol=1e-6)

    def test_pairwise_tanimoto_with_numpy(self):
        """Test that function works with numpy arrays."""
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
        result = pairwise_tanimoto_similarity(a, b)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)

    def test_pairwise_tanimoto_with_list(self):
        """Test that function works with Python lists."""
        a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        b = [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
        result = pairwise_tanimoto_similarity(a, b)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2,)

    def test_pairwise_tanimoto_matches_diagonal_of_full(self):
        """Test that pairwise results match the diagonal of the full similarity matrix."""
        a = torch.randn(5, 10)
        b = torch.randn(5, 10)

        pairwise_result = pairwise_tanimoto_similarity(a, b)
        full_result = tanimoto_similarity(a, b)

        # Extract diagonal
        diagonal = torch.diagonal(full_result)

        assert torch.allclose(pairwise_result, diagonal, atol=1e-6)

    def test_pairwise_tanimoto_zero_handling(self):
        """Test that zero vectors don't cause division by zero."""
        a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        b = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        result = pairwise_tanimoto_similarity(a, b)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))


class TestPairedTanimotoSimilarity:
    """Tests for the paired (NumPy-based) paired_tanimoto_similarity function."""

    def test_paired_tanimoto_identical_vectors(self):
        """Test that identical vectors have paired Tanimoto similarity of 1.0."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = paired_tanimoto_similarity(X, Y)

        assert result.shape == (2,)
        assert np.allclose(result, [1.0, 1.0], atol=1e-6)

    def test_paired_tanimoto_orthogonal_vectors(self):
        """Test that orthogonal vectors have paired Tanimoto similarity of 0.0."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        Y = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = paired_tanimoto_similarity(X, Y)

        assert result.shape == (2,)
        assert np.allclose(result, [0.0, 0.0], atol=1e-6)

    def test_paired_tanimoto_output_shape(self):
        """Test that output has correct shape (n,) for inputs of shape (n, d)."""
        X = np.random.randn(5, 10)
        Y = np.random.randn(5, 10)
        result = paired_tanimoto_similarity(X, Y)

        assert result.shape == (5,)

    def test_paired_tanimoto_range(self):
        """Test that paired Tanimoto similarity is in range [0, 1] for positive vectors."""
        X = np.random.rand(10, 20)
        Y = np.random.rand(10, 20)
        result = paired_tanimoto_similarity(X, Y)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_paired_tanimoto_manual_calculation(self):
        """Test against manual calculation of paired Tanimoto similarity."""
        X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
        Y = np.array([[2.0, 3.0, 4.0], [1.0, 1.0, 1.0]])

        # Manual calculation for first pair
        dot_product_0 = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0  # 20.0
        x_norm_sq_0 = 1.0**2 + 2.0**2 + 3.0**2  # 14.0
        y_norm_sq_0 = 2.0**2 + 3.0**2 + 4.0**2  # 29.0
        expected_0 = dot_product_0 / (x_norm_sq_0 + y_norm_sq_0 - dot_product_0)

        # Manual calculation for second pair
        dot_product_1 = 2.0 * 1.0 + 3.0 * 1.0 + 4.0 * 1.0  # 9.0
        x_norm_sq_1 = 2.0**2 + 3.0**2 + 4.0**2  # 29.0
        y_norm_sq_1 = 1.0**2 + 1.0**2 + 1.0**2  # 3.0
        expected_1 = dot_product_1 / (x_norm_sq_1 + y_norm_sq_1 - dot_product_1)

        result = paired_tanimoto_similarity(X, Y)
        assert np.isclose(result[0], expected_0, atol=1e-6)
        assert np.isclose(result[1], expected_1, atol=1e-6)

    def test_paired_tanimoto_matches_pairwise_torch(self):
        """Test that NumPy paired version matches PyTorch pairwise version."""
        X = np.random.randn(10, 15)
        Y = np.random.randn(10, 15)

        numpy_result = paired_tanimoto_similarity(X, Y)
        torch_result = pairwise_tanimoto_similarity(torch.from_numpy(X).float(), torch.from_numpy(Y).float()).numpy()

        assert np.allclose(numpy_result, torch_result, atol=1e-6)

    def test_paired_tanimoto_array_validation(self):
        """Test that mismatched array shapes raise an error."""
        X = np.array([[1.0, 2.0, 3.0]])
        Y = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises((ValueError, Exception)):
            paired_tanimoto_similarity(X, Y)

    def test_paired_tanimoto_zero_handling(self):
        """Test that zero vectors don't cause division by zero."""
        X = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        Y = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        result = paired_tanimoto_similarity(X, Y)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_paired_tanimoto_with_2d_arrays(self):
        """Test with explicit 2D numpy arrays."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        Y = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]])
        result = paired_tanimoto_similarity(X, Y)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result.dtype in [np.float32, np.float64]


class TestCrossConsistency:
    """Tests to verify consistency between all three implementations."""

    def test_all_functions_on_same_data(self):
        """Test that all three functions produce consistent results on the same paired data."""
        # Create test data
        X_np = np.random.rand(5, 10)
        Y_np = np.random.rand(5, 10)
        X_torch = torch.from_numpy(X_np).float()
        Y_torch = torch.from_numpy(Y_np).float()

        # Compute with all three functions
        paired_result = paired_tanimoto_similarity(X_np, Y_np)
        pairwise_result = pairwise_tanimoto_similarity(X_torch, Y_torch).numpy()
        full_matrix = tanimoto_similarity(X_torch, Y_torch).numpy()
        diagonal_result = np.diagonal(full_matrix)

        # All should be equivalent
        assert np.allclose(paired_result, pairwise_result, atol=1e-6)
        assert np.allclose(paired_result, diagonal_result, atol=1e-6)
        assert np.allclose(pairwise_result, diagonal_result, atol=1e-6)

    def test_properties_hold_across_implementations(self):
        """Test that mathematical properties hold across all implementations."""
        X_np = np.random.rand(3, 5)
        X_torch = torch.from_numpy(X_np).float()

        # Test self-similarity should be 1.0 in all implementations
        paired_self = paired_tanimoto_similarity(X_np, X_np)
        pairwise_self = pairwise_tanimoto_similarity(X_torch, X_torch).numpy()
        full_self_diag = np.diagonal(tanimoto_similarity(X_torch, X_torch).numpy())

        assert np.allclose(paired_self, 1.0, atol=1e-6)
        assert np.allclose(pairwise_self, 1.0, atol=1e-6)
        assert np.allclose(full_self_diag, 1.0, atol=1e-6)
