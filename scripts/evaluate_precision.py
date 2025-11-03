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

"""
Evaluate, benchmark and profile ChemMRL model precision configurations.

This script compares different precision settings (float32, bfloat16, float16) against
ground truth similarity values from the dataset to assess the impact on similarity computations.

Features:
- Accuracy evaluation: Compare precision configurations against dataset ground truth
- Performance benchmarking: Measure inference time, throughput, and memory usage
- CUDA profiling: Track GPU execution time with proper synchronization
- Detailed profiling: Optional PyTorch profiler integration for operation-level analysis
- Warmup runs: Ensure stable performance measurements
- Statistical analysis: Calculate speedup and memory reduction metrics
- Configuration is managed via Hydra (see chem_mrl/conf/evaluate_precision.yaml)

Usage:
    # Basic evaluation (uses config from evaluate_precision.yaml)
    python evaluate_precision.py

    # With detailed PyTorch profiling
    python evaluate_precision.py profile=true

    # With performance benchmarking
    python evaluate_precision.py benchmark=true

    # Both profiling and benchmarking
    python evaluate_precision.py benchmark=true profile=true

    # Custom number of accuracy evaluation samples
    python evaluate_precision.py num_accuracy_samples=10000

    # Custom number of benchmark samples
    python evaluate_precision.py num_benchmark_samples=10000

    # Override model or dataset
    python evaluate_precision.py model_name=Derify/ChemMRL dataset_name=my_dataset

    # Custom output file
    python evaluate_precision.py output_file=my_results.txt
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from transformers.utils.import_utils import is_torch_tf32_available

from chem_mrl import ChemMRL


@dataclass
class PerformanceMetrics:
    """Container for performance profiling metrics."""

    mean_inference_time: float
    std_inference_time: float
    throughput_samples_per_sec: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    peak_memory_mb: float
    cuda_time_ms: float = 0.0


class ChemMRLPrecisionEvaluator:
    """
    Evaluator for ChemMRL model precision configurations.

    This class encapsulates the complete evaluation pipeline for comparing different
    precision settings (float32, bfloat16, float16) and their impact on accuracy and performance.

    Attributes:
        model_name: HuggingFace model identifier
        dataset_name: HuggingFace dataset identifier
        dataset_split: Dataset split to use (train/test/validation)
        batch_size: Batch size for inference
        num_accuracy_samples: Number of samples for accuracy evaluation (None = use all)
        num_benchmark_samples: Number of samples for benchmarking
        num_warmup_runs: Number of warmup iterations
        num_benchmark_runs: Number of benchmark iterations
        enable_profile: Whether to enable detailed PyTorch profiling
        enable_benchmark: Whether to enable performance benchmarking
        tf32_mode: Whether to enable TF32 mode
        test_configs: List of test configurations to evaluate
        smiles_a: First SMILES strings from dataset
        smiles_b: Second SMILES strings from dataset
        ground_truth_similarities: Ground truth similarity values
        output_path: Path to output log file

    Example:
        >>> cfg = OmegaConf.load("config.yaml")
        >>> evaluator = ChemMRLPrecisionEvaluator(cfg)
        >>> results = evaluator.run()
    """

    # Class-level constants
    SEPARATOR = "=" * 90

    def __init__(self, cfg: DictConfig):
        """
        Initialize evaluator with configuration.

        Args:
            cfg: Hydra configuration object containing all evaluation parameters
        """
        # Model and dataset configuration
        self.model_name = cfg.get("model_name", "Derify/ChemMRL")
        self.dataset_name = cfg.get("dataset_name", "Derify/pubchem_10m_genmol_similarity")
        self.dataset_split = cfg.get("dataset_split", "test")

        # Evaluation parameters
        self.batch_size = int(cfg.get("batch_size", 1024))
        self.num_accuracy_samples = cfg.get("num_accuracy_samples", None)
        if self.num_accuracy_samples is not None:
            self.num_accuracy_samples = int(self.num_accuracy_samples)
        self.num_benchmark_samples = int(cfg.get("num_benchmark_samples", 65536))
        self.num_warmup_runs = int(cfg.get("num_warmup_runs", 1))
        self.num_benchmark_runs = int(cfg.get("num_benchmark_runs", 3))

        # Execution flags
        self.enable_profile = bool(cfg.get("profile", False))
        self.enable_benchmark = bool(cfg.get("benchmark", False))
        self.tf32_mode = bool(cfg.get("tf32_mode", True))
        output_file = cfg.get("output_file", None)

        # Test configurations
        test_configs_raw = cfg.get("test_configurations", [])
        if not test_configs_raw:
            raise ValueError("No test_configurations found in config file")
        self.test_configs = OmegaConf.to_container(test_configs_raw, resolve=True)
        if not isinstance(self.test_configs, list):
            raise ValueError("test_configurations must be a list")

        # Data storage (initialized by load_dataset)
        self.smiles_a: list[str] = []
        self.smiles_b: list[str] = []
        self.ground_truth_similarities: np.ndarray = np.array([])

        self._validate_parameters()
        self.output_path = self._setup_logging(output_file)

        if self.tf32_mode:
            if is_torch_tf32_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                logging.warning("TF32 mode requested but current environment does not support TF32.")

    def _validate_parameters(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_accuracy_samples is not None and self.num_accuracy_samples <= 0:
            raise ValueError(f"num_accuracy_samples must be positive or None, got {self.num_accuracy_samples}")
        if self.num_benchmark_samples <= 0:
            raise ValueError(f"num_benchmark_samples must be positive, got {self.num_benchmark_samples}")
        if self.num_warmup_runs < 0:
            raise ValueError(f"num_warmup_runs must be non-negative, got {self.num_warmup_runs}")
        if self.num_benchmark_runs <= 0:
            raise ValueError(f"num_benchmark_runs must be positive, got {self.num_benchmark_runs}")

    @staticmethod
    def _setup_logging(output_file: str | None = None) -> Path:
        """
        Configure logging to both console and file.

        Args:
            output_file: Optional path to save results. If None, uses timestamp.

        Returns:
            Path: The absolute path to the output file.
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"precision_evaluation_{timestamp}.txt"

        output_path = Path(output_file)

        # Configure logging with both file and console handlers
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(output_path, mode="w", encoding="utf-8"),
                logging.StreamHandler(),  # Console output
            ],
            force=True,  # Override any existing configuration
        )

        return output_path

    @staticmethod
    def get_memory_stats() -> dict[str, float]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            "peak_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }

    @staticmethod
    def compute_error_metrics(ground_truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
        """
        Compute comprehensive error metrics between ground truth and predictions.

        Args:
            ground_truth: Ground truth similarity values.
            predictions: Predicted similarity values.

        Returns:
            Dictionary containing various error metrics.
        """
        # Correlation metrics
        spearman_corr, spearman_p = spearmanr(ground_truth, predictions)
        pearson_corr, pearson_p = pearsonr(ground_truth, predictions)

        # Error metrics
        mae = np.mean(np.abs(ground_truth - predictions))
        mse = mean_squared_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        max_error = np.max(np.abs(ground_truth - predictions))
        # Avoid division by zero in MAPE
        non_zero_mask = ground_truth != 0
        if np.any(non_zero_mask):
            mape = mean_absolute_percentage_error(ground_truth[non_zero_mask], predictions[non_zero_mask]) * 100
        else:
            mape = np.nan

        return {
            "spearman_correlation": float(spearman_corr),
            "spearman_p_value": float(spearman_p),
            "pearson_correlation": float(pearson_corr),
            "pearson_p_value": float(pearson_p),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "mape": float(mape),
            "max_error": float(max_error),
        }

    def load_dataset(self):
        """Load and prepare the evaluation dataset."""
        logging.info("\n[1/2] Loading dataset...")
        dataset = load_dataset(self.dataset_name, split=self.dataset_split)
        self.smiles_a = dataset["smiles_a"]
        self.smiles_b = dataset["smiles_b"]
        self.ground_truth_similarities = np.array(dataset["similarity"])

        logging.info(f"✓ Loaded {len(self.smiles_a):,} SMILES pairs with ground truth similarities")
        logging.info(
            f"  Ground truth stats: μ={np.mean(self.ground_truth_similarities):.4f} "
            f"σ={np.std(self.ground_truth_similarities):.4f} "
            f"[{np.min(self.ground_truth_similarities):.4f}, {np.max(self.ground_truth_similarities):.4f}]"
        )
        if self.num_accuracy_samples is not None:
            logging.info(f"  Note: Accuracy evaluation will use {self.num_accuracy_samples:,} samples")
        else:
            logging.info(f"  Note: Accuracy evaluation will use all {len(self.smiles_a):,} samples")
        if self.enable_benchmark or self.enable_profile:
            logging.info(f"  Note: Benchmarking/profiling will use {self.num_benchmark_samples:,} samples")

    def compute_similarities(self, model: ChemMRL, smiles_a: list[str], smiles_b: list[str]) -> np.ndarray:
        """
        Compute pairwise similarities between SMILES pairs.

        Args:
            model: ChemMRL model instance.
            smiles_a: List of first SMILES strings.
            smiles_b: List of second SMILES strings.

        Returns:
            Numpy array of similarity scores.
        """
        embeddings_a = model.embed(
            smiles_a, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False
        )
        embeddings_b = model.embed(
            smiles_b, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False
        )
        similarities: torch.Tensor = model.similarity_pairwise(embeddings_a, embeddings_b)  # pyright: ignore[reportCallIssue, reportArgumentType]
        if similarities.dtype in {torch.bfloat16, torch.float16}:
            similarities = similarities.to(torch.float32)
        if hasattr(similarities, "cpu"):
            return similarities.cpu().numpy()
        return similarities.numpy()

    def benchmark_inference(self, model: ChemMRL, smiles_a: list[str], smiles_b: list[str]) -> PerformanceMetrics:
        """
        Benchmark model inference performance with proper warmup and timing.

        Args:
            model: ChemMRL model instance.
            smiles_a: List of first SMILES strings.
            smiles_b: List of second SMILES strings.

        Returns:
            PerformanceMetrics object with profiling results.
        """
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Warmup phase
        for _ in range(self.num_warmup_runs):
            embeddings_a = model.embed(
                smiles_a, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False
            )
            embeddings_b = model.embed(
                smiles_b, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False
            )
            _ = model.similarity_pairwise(embeddings_a, embeddings_b)  # pyright: ignore[reportCallIssue, reportArgumentType]
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Benchmark phase
        timings = []
        cuda_timings = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for _ in range(self.num_benchmark_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            start_time = time.perf_counter()

            embeddings_a = model.embed(
                smiles_a, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False
            )
            embeddings_b = model.embed(
                smiles_b, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False
            )
            _ = model.similarity_pairwise(embeddings_a, embeddings_b)  # pyright: ignore[reportCallIssue, reportArgumentType]

            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                cuda_time = start_event.elapsed_time(end_event)
                cuda_timings.append(cuda_time)

            end_time = time.perf_counter()
            timings.append(end_time - start_time)

        # Calculate statistics
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        throughput = len(smiles_a) / mean_time

        memory_stats = self.get_memory_stats()
        cuda_time_ms = np.mean(cuda_timings) if cuda_timings else 0.0

        return PerformanceMetrics(
            mean_inference_time=float(mean_time),
            std_inference_time=float(std_time),
            throughput_samples_per_sec=float(throughput),
            memory_allocated_mb=memory_stats["allocated_mb"],
            memory_reserved_mb=memory_stats["reserved_mb"],
            peak_memory_mb=memory_stats["peak_mb"],
            cuda_time_ms=float(cuda_time_ms),
        )

    def detailed_profiling(self, model: ChemMRL, smiles_a: list[str], smiles_b: list[str]) -> str:
        """
        Run detailed PyTorch profiling and return formatted results.

        Args:
            model: ChemMRL model instance.
            smiles_a: List of first SMILES strings.
            smiles_b: List of second SMILES strings.

        Returns:
            Formatted profiling results as string.
        """
        if not torch.cuda.is_available():
            return "Detailed profiling requires CUDA"

        with torch_profiler_context(enabled=True) as prof:
            embeddings_a = model.embed(
                smiles_a, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False
            )
            embeddings_b = model.embed(
                smiles_b, batch_size=self.batch_size, convert_to_tensor=True, show_progress_bar=False
            )
            _ = model.similarity_pairwise(embeddings_a, embeddings_b)  # pyright: ignore[reportCallIssue, reportArgumentType]

        if prof is None:
            return "Profiling not available"

        # Get profiler results
        profile_output = []
        profile_output.append("\n" + self.SEPARATOR)
        profile_output.append("DETAILED PROFILING RESULTS")
        profile_output.append(self.SEPARATOR)

        # Key operations by time
        row_limit = 14
        profile_output.append("\nTop operations by CUDA time:")
        profile_output.append(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=row_limit,
                max_src_column_width=84,
                max_name_column_width=64,
                max_shapes_column_width=0,
                header=f"Top {row_limit} operations by CUDA time",
            )
        )

        # Memory operations
        profile_output.append("\nTop operations by memory:")
        profile_output.append(
            prof.key_averages().table(
                sort_by="self_cuda_memory_usage",
                row_limit=row_limit,
                max_name_column_width=64,
                max_shapes_column_width=0,
                header=f"Top {row_limit} operations by memory usage",
            )
        )

        return "\n".join(profile_output)

    def print_config_results(
        self, name: str, similarities: np.ndarray, metrics: dict[str, float], perf: PerformanceMetrics | None = None
    ):
        """Print consolidated results for a configuration."""
        logging.info(f"\n{name}")
        logging.info("-" * 80)

        # Similarity stats
        logging.info(
            f"  Similarity: μ={np.mean(similarities):.4f} σ={np.std(similarities):.4f} "
            f"[{np.min(similarities):.4f}, {np.max(similarities):.4f}]"
        )

        # Accuracy metrics - all error metrics
        logging.info(
            f"  Accuracy:   Spearman={metrics['spearman_correlation']:.6f} Pearson={metrics['pearson_correlation']:.6f}"
        )
        logging.info(
            f"              MAE={metrics['mae']:.6f} MSE={metrics['mse']:.6f} RMSE={metrics['rmse']:.6f} "
            f"Max={metrics['max_error']:.6f}"
        )
        if not np.isnan(metrics["mape"]):
            logging.info(f"              MAPE={metrics['mape']:.4f}%")

        # Performance metrics
        if perf:
            perf_msg = (
                f"  Performance: {perf.mean_inference_time:.3f}s "
                f"({perf.throughput_samples_per_sec:.1f} samples/s) "
                f"Peak Mem={perf.peak_memory_mb:.1f}MB"
            )
            if perf.cuda_time_ms > 0:
                perf_msg += f" CUDA={perf.cuda_time_ms:.1f}ms"
            logging.info(perf_msg)

    def print_summary_table(self, results: list[dict[str, Any]]):
        """Print a comprehensive summary table of all results."""
        logging.info("\n\n" + self.SEPARATOR)
        logging.info("ACCURACY SUMMARY")
        logging.info(self.SEPARATOR)

        if not results:
            logging.info("No results to display")
            return

        # Get baseline metrics (float32 default)
        baseline_metrics = results[0]["metrics"]

        # Compact table header - split into multiple sections for readability
        logging.info(f"{'Configuration':<42} {'Spearman':>12} {'Δ%':>9} {'Pearson':>12} {'Δ%':>9}")
        logging.info("-" * 92)

        # Data rows - Correlation metrics
        for result in results:
            config = result["config"]
            metrics = result["metrics"]

            spearman_delta = (
                (metrics["spearman_correlation"] - baseline_metrics["spearman_correlation"])
                / abs(baseline_metrics["spearman_correlation"])
                * 100
                if baseline_metrics["spearman_correlation"] != 0
                else 0.0
            )
            pearson_delta = (
                (metrics["pearson_correlation"] - baseline_metrics["pearson_correlation"])
                / abs(baseline_metrics["pearson_correlation"])
                * 100
                if baseline_metrics["pearson_correlation"] != 0
                else 0.0
            )

            row = (
                f"{config:<42} "
                f"{metrics['spearman_correlation']:>12.6f} "
                f"{spearman_delta:>8.2f}% "
                f"{metrics['pearson_correlation']:>12.6f} "
                f"{pearson_delta:>8.2f}%"
            )
            logging.info(row)

        logging.info("")
        logging.info(f"{'Configuration':<42} {'MAE':>12} {'Δ%':>9} {'MSE':>12} {'Δ%':>9}")
        logging.info("-" * 92)

        # Data rows - Error metrics part 1
        for result in results:
            config = result["config"]
            metrics = result["metrics"]

            mae_delta = (
                (metrics["mae"] - baseline_metrics["mae"]) / baseline_metrics["mae"] * 100
                if baseline_metrics["mae"] != 0
                else 0.0
            )
            mse_delta = (
                (metrics["mse"] - baseline_metrics["mse"]) / baseline_metrics["mse"] * 100
                if baseline_metrics["mse"] != 0
                else 0.0
            )

            row = f"{config:<42} {metrics['mae']:>12.6f} {mae_delta:>8.2f}% {metrics['mse']:>12.6f} {mse_delta:>8.2f}%"
            logging.info(row)

        logging.info("")
        logging.info(f"{'Configuration':<42} {'RMSE':>12} {'Δ%':>9} {'Max Error':>12} {'Δ%':>9}")
        logging.info("-" * 92)

        # Data rows - Error metrics part 2
        for result in results:
            config = result["config"]
            metrics = result["metrics"]

            rmse_delta = (
                (metrics["rmse"] - baseline_metrics["rmse"]) / baseline_metrics["rmse"] * 100
                if baseline_metrics["rmse"] != 0
                else 0.0
            )
            max_error_delta = (
                (metrics["max_error"] - baseline_metrics["max_error"]) / baseline_metrics["max_error"] * 100
                if baseline_metrics["max_error"] != 0
                else 0.0
            )

            row = (
                f"{config:<42} "
                f"{metrics['rmse']:>12.6f} "
                f"{rmse_delta:>8.2f}% "
                f"{metrics['max_error']:>12.6f} "
                f"{max_error_delta:>8.2f}%"
            )
            logging.info(row)

        # Performance metrics table
        logging.info("\n\n" + self.SEPARATOR)
        logging.info("PERFORMANCE SUMMARY")
        logging.info(self.SEPARATOR)

        perf_header = f"{'Configuration':<42} {'Time(s)':>10} {'Speedup':>11} {'Mem(MB)':>10} {'Mem %':>11}"
        logging.info(perf_header)
        logging.info("-" * 92)

        if results and "performance" in results[0]:
            baseline_time = results[0]["performance"].mean_inference_time
            baseline_memory = results[0]["performance"].peak_memory_mb

            for result in results:
                config = result["config"]
                if "performance" in result:
                    perf = result["performance"]
                    speedup = baseline_time / perf.mean_inference_time
                    memory_reduction = (1 - perf.peak_memory_mb / baseline_memory) * 100 if baseline_memory > 0 else 0

                    perf_row = (
                        f"{config:<42} "
                        f"{perf.mean_inference_time:>10.4f} "
                        f"{speedup:>9.2f}x "
                        f"{perf.peak_memory_mb:>10.1f} "
                        f"{memory_reduction:>9.1f}%"
                    )
                    logging.info(perf_row)

        logging.info(self.SEPARATOR)

    def evaluate_configuration(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate a single model configuration.

        Args:
            config: Configuration dictionary with name and model parameters.

        Returns:
            Dictionary containing configuration name, metrics, and performance data.
        """
        model_init_kwargs = {k: v for k, v in config.items() if k != "name"}
        model = ChemMRL(**model_init_kwargs)

        # Embedding similarity evaluation
        accuracy_smiles_a = self.smiles_a
        accuracy_smiles_b = self.smiles_b
        accuracy_ground_truth = self.ground_truth_similarities
        if self.num_accuracy_samples is not None:
            accuracy_indices = np.random.choice(
                len(self.smiles_a), size=min(self.num_accuracy_samples, len(self.smiles_a)), replace=False
            )
            accuracy_smiles_a = [self.smiles_a[i] for i in accuracy_indices]
            accuracy_smiles_b = [self.smiles_b[i] for i in accuracy_indices]
            accuracy_ground_truth = self.ground_truth_similarities[accuracy_indices]

        similarities = self.compute_similarities(model, accuracy_smiles_a, accuracy_smiles_b)
        metrics = self.compute_error_metrics(accuracy_ground_truth, similarities)

        result = {
            "config": config["name"],
            "metrics": metrics,
        }

        # Benchmarking and profiling: use random SUBSET of samples
        indices = np.random.choice(
            len(self.smiles_a), size=min(self.num_benchmark_samples, len(self.smiles_a)), replace=False
        )
        benchmark_smiles_a = [self.smiles_a[i] for i in indices]
        benchmark_smiles_b = [self.smiles_b[i] for i in indices]

        if self.enable_benchmark:
            perf_metrics = self.benchmark_inference(model, benchmark_smiles_a, benchmark_smiles_b)
            result["performance"] = perf_metrics
            self.print_config_results(config["name"], similarities, metrics, perf_metrics)
        else:
            self.print_config_results(config["name"], similarities, metrics)

        if self.enable_profile:
            profile_output = self.detailed_profiling(model, benchmark_smiles_a, benchmark_smiles_b)
            logging.info(profile_output)

        return result

    def run(self) -> list[dict[str, Any]]:
        """
        Execute the complete evaluation pipeline.

        This is the main entry point for running the evaluation. It performs:
        1. Prints configuration summary
        2. Loads the dataset
        3. Evaluates all test configurations
        4. Prints comprehensive summary tables
        5. Saves results to file

        Returns:
            List of result dictionaries, one per configuration.
        """
        logging.info(self.SEPARATOR)
        logging.info(f"ChemMRL Precision Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Dataset: {self.dataset_name} ({self.dataset_split})")
        if self.enable_benchmark or self.enable_profile:
            logging.info(f"Benchmark samples: {self.num_benchmark_samples:,}")
        logging.info(f"Warmup runs: {self.num_warmup_runs}, Benchmark runs: {self.num_benchmark_runs}")
        logging.info(self.SEPARATOR)

        self.load_dataset()

        logging.info("\n[2/2] Evaluating precision configurations...")
        results = []
        for i, config in enumerate(self.test_configs, 1):
            logging.info(f"\n  [{i}/{len(self.test_configs)}] {config['name']}...")
            result = self.evaluate_configuration(config)
            results.append(result)

        self.print_summary_table(results)
        logging.info(f"\n✓ Results saved to: {self.output_path.absolute()}")

        return results


@contextmanager
def torch_profiler_context(enabled: bool = True):
    """Context manager for PyTorch profiling with CUDA synchronization."""
    if not enabled or not torch.cuda.is_available():
        yield None
        return

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        yield prof
    torch.cuda.synchronize()


@hydra.main(
    config_path="../chem_mrl/conf",
    config_name="evaluate_precision",
    version_base="1.2",
)
def main(cfg: DictConfig):
    """
    Main entry point for the evaluation script.

    Creates an evaluator instance from the Hydra configuration and runs the evaluation pipeline.

    Args:
        cfg: Hydra configuration object loaded from YAML config file.
    """
    evaluator = ChemMRLPrecisionEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
