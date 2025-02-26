from dataclasses import dataclass
from typing import Literal, TypedDict


@dataclass
class GenMolInferenceConfig:
    num_molecules: int  # Number of Molecules to Generate
    temperature: float  # Temperature Scaling Factor for Softmax Sampling
    noise: float  # Noise Factor for Top-K Sampling
    step_size: int  # Diffusion Step Size
    unique: bool  # Show only unique molecules
    scoring: Literal["QED", "LogP"]  # Scoring Method

    def __post_init__(self):
        if self.num_molecules < 1 or self.num_molecules > 1000:
            raise ValueError(f"num_molecules must be between 1 and 1000, got {self.num_molecules}.")
        if self.temperature < 0.01 or self.temperature > 10.0:
            raise ValueError(f"temperature must be between 0.01 and 10.0, got {self.temperature}.")
        if self.noise < 0.0 or self.noise > 2.0:
            raise ValueError(f"noise must be between 0.0 and 2.0, got {self.noise}.")
        if self.step_size < 1 or self.step_size > 10:
            raise ValueError(f"step_size must be between 1 and 10, got {self.step_size}.")
        if self.unique not in [True, False]:
            raise ValueError(f"unique must be True or False, got {self.unique}.")
        if self.scoring not in ["QED", "LogP"]:
            raise ValueError(f"Invalid scoring method: {self.scoring}. Must be 'QED' or 'LogP'.")


@dataclass
class GenMolProduceConfig:
    min_tokens_to_generate: int
    max_tokens_to_generate: int
    num_unique_generations: int
    num_top_scored_molecules_to_keep: int
    inference: GenMolInferenceConfig
    invoke_urls: list[str]
    num_workers: int = 1


@dataclass
class DatasetConfig:
    path: str
    batch_size: int
    number_of_samples_to_fragment: int
    num_workers: int
    generate_scores: bool
    smiles_column: str
    safe_column: str | None
    score_column: str | None


@dataclass
class ScoreConfig:
    score: Literal["QED", "SA", "LogP"]
    score_cutoff: float = 0.9
    similarity_cutoff: float = 0.7
    fingerprint_type: Literal["morgan", "functional"] = "morgan"
    use_composite_score: bool = False


@dataclass
class EvalDatasetDict(TypedDict):
    smiles: str
    base_score: float
    score: float
    similarity: float
    reference_smiles: str
    reference_score: float | None
