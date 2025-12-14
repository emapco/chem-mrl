import pandas as pd
import pytest

from ..genmol import GenMolGenerator
from ..types import GenMolInferenceConfig, GenMolProduceConfig


@pytest.fixture
def generator():
    return GenMolGenerator()


@pytest.fixture
def molecules():
    return pd.Series([f"mol_{i}" for i in range(100)])


@pytest.fixture
def base_config():
    inference_config = GenMolInferenceConfig(
        num_molecules=1, temperature=1.0, noise=0.0, step_size=1, unique=True, scoring="QED"
    )
    return GenMolProduceConfig(
        min_tokens_to_generate=1,
        max_tokens_to_generate=5,
        num_unique_generations=1,
        num_top_scored_molecules_to_keep=1,
        inference=inference_config,
        invoke_urls=[],
        api_instances_per_partition=[],
        partition_fractions=[],
    )


class TestGenMolPartitioning:
    def test_valid_partitions(self, generator, molecules, base_config):
        # 2 partitions: 40% and 60%
        # Partition 1: 2 instances (total 40 items -> 20 each)
        # Partition 2: 1 instance (total 60 items -> 60 each)
        base_config.api_instances_per_partition = [2, 1]
        base_config.partition_fractions = [0.4, 0.6]
        base_config.invoke_urls = ["url1", "url2", "url3"]  # 2 + 1 = 3 urls

        partitions = generator.generate_partitions(molecules, base_config)

        assert len(partitions) == 3  # 2 + 1 sub-partitions
        assert len(partitions[0]) == 20
        assert len(partitions[1]) == 20
        assert len(partitions[2]) == 60

        # Verify content coverage
        all_mols = []
        for p in partitions:
            all_mols.extend(p)
        assert len(all_mols) == 100
        assert set(all_mols) == set(molecules)

    def test_invalid_fractions_sum(self, generator, molecules, base_config):
        base_config.api_instances_per_partition = [1, 1]
        base_config.partition_fractions = [0.4, 0.5]  # Sums to 0.9
        base_config.invoke_urls = ["url1", "url2"]

        with pytest.raises(ValueError, match="Partition fractions must sum to 1.0"):
            generator.generate_partitions(molecules, base_config)

    def test_mismatched_lengths(self, generator, molecules, base_config):
        base_config.api_instances_per_partition = [1, 1]
        base_config.partition_fractions = [1.0]  # Mismatched length
        base_config.invoke_urls = ["url1", "url2"]

        with pytest.raises(ValueError, match="must match number of partitions"):
            generator.generate_partitions(molecules, base_config)

    def test_mismatched_urls(self, generator, molecules, base_config):
        base_config.api_instances_per_partition = [1, 1]
        base_config.partition_fractions = [0.5, 0.5]
        base_config.invoke_urls = ["url1"]  # Should be 2

        with pytest.raises(ValueError, match="must match total API instances"):
            generator.generate_partitions(molecules, base_config)

    def test_single_partition(self, generator, molecules, base_config):
        base_config.api_instances_per_partition = [2]
        base_config.partition_fractions = [1.0]
        base_config.invoke_urls = ["url1", "url2"]

        partitions = generator.generate_partitions(molecules, base_config)

        assert len(partitions) == 2
        assert len(partitions[0]) == 50
        assert len(partitions[1]) == 50

    def test_zero_partitions(self, generator, molecules, base_config):
        base_config.api_instances_per_partition = []
        base_config.partition_fractions = []
        base_config.invoke_urls = []

        partitions = generator.generate_partitions(molecules, base_config)
        assert partitions is None
