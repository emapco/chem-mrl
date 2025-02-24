import argparse
import os

import numpy as np
import pandas as pd
import torch

# this script is meant to be run in bionemo-framework-1 (or molmim) docker container
from bionemo.model.core.controlled_generation import (
    ControlledGenerationPerceiverEncoderInferenceWrapper,
)
from bionemo.model.molecule.molmim.infer import MolMIMInference
from bionemo.utils.hydra import load_model_config
from guided_molecule_gen.optimizer import MoleculeGenerationOptimizer
from guided_molecule_gen.oracles import qed, tanimoto_similarity
from tqdm import trange
from util import canonicalize_smiles, init_logging

logger = init_logging(__name__)


def configure_pytorch():
    """Configures PyTorch settings."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Molecule Optimization Pipeline for Smiles Dataset Augmentation"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset file. Output file will have `_molmim_augmented` append to current file name",
    )
    parser.add_argument(
        "--smiles_column_name",
        type=str,
        default="canonical_smiles",
        help="SMILES column name. For optimal performance, ensure smiles are RDkit canonicalized.",
    )
    parser.add_argument("--qed_cutoff", type=float, default=0.875, help="QED cutoff value")
    parser.add_argument(
        "--tanimoto_cutoff", type=float, default=0.6, help="Tanimoto similarity cutoff"
    )
    parser.add_argument(
        "--beam_size", type=int, default=3, help="Beam size for molecule generation"
    )
    parser.add_argument("--optimizer_steps", type=int, default=5, help="Number of optimizer steps")
    parser.add_argument(
        "--optimizer_sigma", type=float, default=0.438864, help="Optimizer sigma value"
    )
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size")
    parser.add_argument(
        "--optimizer_popsize", type=int, default=10, help="Optimizer population size"
    )
    return parser.parse_args()


def get_controlled_generation_model(beam_size: int):
    """Initializes and returns the controlled generation model."""
    controlled_gen_kwargs = {
        "sampling_method": "beam-search",
        "sampling_kwarg_overrides": {
            "beam_size": beam_size,
            "keep_only_best_tokens": True,
            "return_scores": False,
        },
    }
    bionemo_home = "/workspace/bionemo"
    os.environ["BIONEMO_HOME"] = bionemo_home
    checkpoint_path = f"{bionemo_home}/chem-mrl/models/molmim_70m_24_3.nemo"
    cfg = load_model_config(
        config_name="molmim_infer.yaml",
        config_path=f"{bionemo_home}/examples/tests/conf/",
    )
    cfg.model.downstream_task.restore_from_path = checkpoint_path
    model = MolMIMInference(cfg, interactive=True)
    return ControlledGenerationPerceiverEncoderInferenceWrapper(
        model, enforce_perceiver=True, hidden_steps=1, **controlled_gen_kwargs
    )


def scoring_function_closure_factory(qed_cutoff, tanimoto_cutoff):
    """Creates a scoring function closure for molecule optimization."""

    def score_mixing_function(qeds, similarities):
        return np.clip(qeds / qed_cutoff, 0.0, 1.0) + np.clip(
            similarities / tanimoto_cutoff, 0.0, 1.0
        )

    def scoring_function(smiles: list[str], reference: str, **kwargs) -> np.ndarray:
        smiles = [canonicalize_smiles(smi) or "" for smi in smiles]
        reference = canonicalize_smiles(reference) or ""
        return -1 * score_mixing_function(qed(smiles), tanimoto_similarity(smiles, reference))

    return scoring_function


def main():
    ARGS = parse_args()
    configure_pytorch()

    df = pd.read_parquet(ARGS.dataset_path, columns=[ARGS.smiles_column_name])
    output_path = ARGS.dataset_path.replace(".parquet", "_molmim_augmented.parquet")
    if os.path.exists(output_path):
        sampled_df = pd.read_parquet(output_path)
        sampled_df = sampled_df.astype(
            {
                "reference_smiles": "category",
                "batch_start": "category",
                "qed_score": np.float16,
                "tanimoto_score": np.float16,
                "reference_qed_score": np.float16,
            }
        )
        last_row_batch_start = sampled_df["batch_start"].iloc[-1]
        resume_batch_start = int(last_row_batch_start) + ARGS.batch_size
        logger.info(f"Resuming from batch {resume_batch_start}")
    else:
        sampled_df = pd.DataFrame()
        resume_batch_start = 0
        logger.info("Starting a new job")

    model = get_controlled_generation_model(ARGS.beam_size)
    scoring_function = scoring_function_closure_factory(ARGS.qed_cutoff, ARGS.tanimoto_cutoff)

    for batch_start in trange(
        resume_batch_start, len(df), ARGS.batch_size, desc="Processing Batches"
    ):
        batch_end = batch_start + ARGS.batch_size
        batch_smiles = df[ARGS.smiles_column_name].iloc[batch_start:batch_end].to_list()
        sampled_data = []
        optimizer = MoleculeGenerationOptimizer(
            model,
            scoring_function,
            batch_smiles,
            popsize=ARGS.optimizer_popsize,
            optimizer_args={"sigma": ARGS.optimizer_sigma},
        )

        for _ in trange(ARGS.optimizer_steps):
            optimizer.step()
            final_smiles = optimizer.generated_smis
            scores = np.array(
                [scoring_function(smis, ref) for smis, ref in zip(final_smiles, batch_smiles)]
            )
            best_indices = np.argmin(scores, axis=1)
            best_smis = [smis[idx] for smis, idx in zip(final_smiles, best_indices)]
            canonicalized_best_molecules = [canonicalize_smiles(smis) for smis in best_smis]
            qed_scores = qed(best_smis)
            tanimoto_scores = [
                tanimoto_similarity([smis], ref)[0] for smis, ref in zip(best_smis, batch_smiles)
            ]

            # only store molecules that have changed in the current step
            for smi, can_smi, qed_score, tanimoto_score, ref_smi, ref_qed in zip(
                best_smis,
                canonicalized_best_molecules,
                qed_scores,
                tanimoto_scores,
                batch_smiles,
                qed(batch_smiles),
            ):
                if tanimoto_score != 1.0:
                    sampled_data.append(
                        {
                            "smiles": smi,
                            "canonical_smiles": can_smi,
                            "qed_score": np.float16(qed_score),
                            "tanimoto_score": np.float16(tanimoto_score),
                            "reference_smiles": ref_smi,
                            "reference_qed_score": np.float16(ref_qed),
                            "batch_start": str(batch_start),
                        }
                    )

        batch_df = pd.DataFrame(sampled_data)
        batch_df["reference_smiles"] = batch_df["reference_smiles"].astype("category")
        batch_df["batch_start"] = batch_df["batch_start"].astype("category")
        sampled_df = pd.concat([sampled_df, batch_df], ignore_index=True)
        logger.info(f"Saving... {batch_start}-{batch_end}\n")
        sampled_df.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")

    sampled_df.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")


if __name__ == "__main__":
    main()
