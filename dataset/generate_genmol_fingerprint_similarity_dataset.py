import os

import hydra
import numpy as np
import pandas as pd
from genmol.genmol import GenMolGenerator
from genmol.library import BaseLibrary
from genmol.oracle import Oracle
from genmol.types import DatasetConfig, EvalDatasetDict, GenMolProduceConfig, ScoreConfig
from genmol.utils import Utils
from omegaconf import DictConfig
from pandarallel import pandarallel
from tqdm import tqdm, trange
from util import init_logging

logger = init_logging(__name__)


class FingerprintDatasetGenerator:
    __fingerprint_dataset_columns = [*list(EvalDatasetDict.__required_keys__), "batch_start"]
    __similarity_dataset_types = {
        "base_score": np.float32,
        "score": np.float32,
        "similarity": np.float32,
        "reference_smiles": "category",
        "reference_score": np.float32,
        "batch_start": "category",
    }

    def __init__(self, cfg: DictConfig) -> None:
        self.seed = cfg.seed
        self.gen_config: GenMolProduceConfig = hydra.utils.instantiate(cfg.generator)
        self.dataset_cfg: DatasetConfig = hydra.utils.instantiate(cfg.dataset)
        self.score_cfg: ScoreConfig = hydra.utils.instantiate(cfg.oracle)

        oracle_score = Oracle.RDKitScore(name=self.score_cfg.score, score_config=self.score_cfg)
        self.oracle = Oracle(score=oracle_score)
        self.generator = GenMolGenerator()
        # column names for loadings datasets
        self.smiles_column = self.dataset_cfg.smiles_column
        self.score_column = self.dataset_cfg.score_column
        self.oracle_name = self.score_cfg.score.lower()
        self.safe_column = self.dataset_cfg.safe_column or "safe"

        if self.dataset_cfg.num_workers is not None:
            pandarallel.initialize(progress_bar=True, nb_workers=self.dataset_cfg.num_workers)
        else:
            pandarallel.initialize(progress_bar=True)

    def _init_fragments(self, mol_df: pd.DataFrame):
        oracle_score = Oracle(Oracle.RDKitScore(name=self.score_cfg.score))
        self.fragments = BaseLibrary.fragment_molecules(
            mol_df["smiles"].sample(
                self.dataset_cfg.number_of_samples_to_fragment, random_state=self.seed
            ),
            oracle_score,
            self.dataset_cfg.num_workers,
        )

    def _format_mol_df(
        self, df: pd.DataFrame, safe_column: str, score_column: str | None
    ) -> pd.DataFrame:
        # remap column names
        rename_map = {
            self.dataset_cfg.smiles_column: "smiles",
            safe_column: "safe",
        }
        if self.dataset_cfg.generate_scores and score_column is not None:
            rename_map[score_column] = "score"
        df.rename(columns=rename_map, inplace=True)

        logger.info(f"Loaded {len(df)} molecules")
        return df

    def _load_cached_dataset(self):
        # create a new dataset with required data
        output_file_suffix = (
            f"{self.oracle_name}_scores" if self.dataset_cfg.generate_scores else "safe"
        )
        cache_path = self.dataset_cfg.path.replace(".parquet", f"_{output_file_suffix}.parquet")
        if os.path.exists(cache_path):
            logger.info(f"Loading cached safe dataset from {cache_path}")
            df = BaseLibrary.from_parquet(cache_path, columns=[self.smiles_column, "safe"])
            return self._format_mol_df(df, self.safe_column, self.oracle_name)

        logger.info(f"Creating a new dataset from {self.dataset_cfg.path}")
        df = BaseLibrary.from_parquet(self.dataset_cfg.path, columns=[self.smiles_column])

        df = self._generate_score_column(df)
        df = self._generate_safe_column(df)

        df.to_parquet(cache_path, index=False, engine="pyarrow", compression="zstd")
        return self._format_mol_df(df, self.safe_column, self.oracle_name)

    def _generate_safe_column(self, df: pd.DataFrame):
        if self.dataset_cfg.safe_column is None and self.safe_column not in df.columns:
            logger.info("Generating SAFE representations")
            df[self.safe_column] = df[self.dataset_cfg.smiles_column].parallel_apply(
                Utils.smiles2safe
            )
            df.dropna(subset=[self.safe_column], inplace=True)
        return df

    def _generate_score_column(self, df: pd.DataFrame):
        if self.dataset_cfg.generate_scores:
            logger.info(f"Generating {self.oracle_name} scores")
            oracle_score = Oracle.RDKitScore(self.oracle_name)
            df[self.oracle_name] = df[self.smiles_column].parallel_apply(oracle_score)
            df.dropna(subset=[self.oracle_name], inplace=True)
        return df

    def load_smiles_dataset(self) -> pd.DataFrame:
        safe_column = self.dataset_cfg.safe_column  # different logic required here

        columns = [self.smiles_column]
        if self.score_column is not None:
            columns.append(self.score_column)
        if safe_column is not None:
            columns.append(safe_column)

        if len(columns) > 1 and safe_column is not None:
            logger.info(f"Loading dataset from {self.dataset_cfg.path}")
            df = BaseLibrary.from_parquet(self.dataset_cfg.path, columns=columns)
            return self._format_mol_df(df, safe_column, self.score_column)

        return self._load_cached_dataset()

    def load_similarity_dataset(self) -> tuple[pd.DataFrame, int]:
        parent_path = os.path.dirname(os.path.dirname(self.dataset_cfg.path))

        similarity_output_path = os.path.join(parent_path, "fp_genmol")
        similarity_file_name = os.path.basename(self.dataset_cfg.path).replace(
            ".parquet", "_genmol_augmented.parquet"
        )
        os.makedirs(similarity_output_path, exist_ok=True)

        self.similarity_output_file = os.path.join(similarity_output_path, similarity_file_name)

        if os.path.exists(self.similarity_output_file):
            sampled_df = pd.read_parquet(self.similarity_output_file)
            sampled_df = sampled_df.astype(self.__similarity_dataset_types)
            last_batch = sampled_df["batch_start"].iloc[-1]
            resume_batch_start = int(last_batch) + self.dataset_cfg.batch_size
            logger.info(f"Resuming from batch {resume_batch_start}")
        else:
            sampled_df = pd.DataFrame()
            resume_batch_start = 0
            logger.info("Starting a new job")
        return sampled_df, resume_batch_start

    def save_similarity_dataset(self, batch_df: pd.DataFrame, similarity_df: pd.DataFrame):
        batch_df = batch_df.astype(self.__similarity_dataset_types)
        similarity_df = pd.concat([similarity_df, batch_df], ignore_index=True)
        similarity_df.to_parquet(
            self.similarity_output_file, index=False, engine="pyarrow", compression="zstd"
        )
        return similarity_df

    def generate_batch_data(self, safe_iterable: pd.Series, batch_start: int) -> pd.DataFrame:
        batch_smiles = self.generator.produce_similar_smiles(safe_iterable, self.gen_config)

        dfs = []
        for reference, similar_smiles in tqdm(
            batch_smiles.items(), desc="Evaluating generated smiles"
        ):
            eval: list[EvalDatasetDict] = self.oracle.evaluate_for_dataset_gen(
                similar_smiles,
                reference=reference,
                reference_score=None,
            )
            df = pd.DataFrame(eval, columns=self.__fingerprint_dataset_columns)
            df.sort_values(by="score", ascending=False, inplace=True)
            df.dropna(subset="score", inplace=True, ignore_index=True)
            num_keep = min(self.gen_config.num_top_scored_molecules_to_keep, len(df) - 1)
            dfs.append(df.iloc[:num_keep])
        batch_df = pd.concat(dfs)
        batch_df["batch_start"] = batch_start
        return batch_df


@hydra.main(
    config_path="../chem_mrl/conf",
    config_name="genmol",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    logger.info("Starting Fingerprint Dataset Generation")
    dataset_gen = FingerprintDatasetGenerator(cfg)

    mol_df = dataset_gen.load_smiles_dataset()
    similarity_df, resume_batch_start = dataset_gen.load_similarity_dataset()

    batch_size = dataset_gen.dataset_cfg.batch_size
    for batch_start in trange(
        resume_batch_start, len(mol_df), batch_size, desc="Processing Batches"
    ):
        batch_end = min(batch_start + batch_size, len(mol_df) - 1)
        batch_df = mol_df.iloc[batch_start:batch_end]
        batch_data = dataset_gen.generate_batch_data(batch_df["safe"], batch_start)
        logger.info(f"Saving... {batch_start}-{batch_end}\n")
        similarity_df = dataset_gen.save_similarity_dataset(batch_data, similarity_df)

    logger.info("Done!")


if __name__ == "__main__":
    main()
