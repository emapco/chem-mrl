from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor

import hydra
import numpy as np
import pandas as pd
from requests import Session
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util import Retry

from .types import GenMolProduceConfig


class GenMolGenerator:
    __default_params__ = {
        "num_molecules": 10,
        "temperature": 1.0,
        "noise": 0.0,
        "step_size": 1,
        "unique": True,
        "scoring": "QED",
    }

    def __init__(self, invoke_url="http://127.0.0.1:8000/generate", auth=None, **kwargs):
        self.invoke_url = invoke_url
        self.auth = auth
        self.session = Session()
        self.num_generate = kwargs.get("num_generate", 1)
        self.verbose = False
        self.max_retries = kwargs.get("max_retries", 5)
        self.retries = Retry(
            total=self.max_retries,
            backoff_factor=0.1,
            status_forcelist=[400],
            allowed_methods={"POST"},
        )
        self.headers = {
            "Authorization": "" if self.auth is None else "Bearer " + self.auth,
            "Content-Type": "application/json",
        }
        self.session.mount(self.invoke_url, HTTPAdapter(max_retries=self.retries))

    def produce(self, molecules, num_generate):
        generated = []

        for m in molecules:
            safe_segs = m.split(".")
            pos = np.random.randint(len(safe_segs))
            start_seg = len(safe_segs[pos])
            safe_segs[pos] = "[*{%d-%d}]" % (start_seg, start_seg + 5)  # noqa: UP031
            smiles = ".".join(safe_segs)

            new_molecules = self.inference(
                smiles=smiles, num_molecules=max(10, num_generate), temperature=1.5, noise=2.0
            )

            new_molecules = [_["smiles"] for _ in new_molecules]

            if len(new_molecules) == 0:
                return []

            new_molecules = new_molecules[: (min(self.num_generate, len(new_molecules)))]
            generated.extend(new_molecules)

        self.molecules = list(set(generated))
        return self.molecules

    @staticmethod
    def _process_partition(
        molecules_partition: Iterable[str],
        cfg: GenMolProduceConfig,
        inference_fn: Callable,
        invoke_url: str,
    ) -> dict[str, list[str]]:
        """Process a single partition of molecules and generate similar SMILES."""
        inference_cfg = hydra.utils.instantiate(cfg.inference)
        partition_results = {}
        for reference in molecules_partition:
            similar_molecules = []
            for _ in range(cfg.num_unique_generations):
                min_tokens = cfg.min_tokens_to_generate
                max_tokens = cfg.max_tokens_to_generate

                num_tokens = (
                    max_tokens
                    if min_tokens > max_tokens
                    else min_tokens
                    if min_tokens == max_tokens
                    else np.random.randint(min_tokens, max_tokens)
                )

                safe_segs = reference.split(".")
                seg_position_to_mask = np.random.randint(len(safe_segs))
                start_seg = len(safe_segs[seg_position_to_mask])
                safe_segs[seg_position_to_mask] = f"[*{{{start_seg}-{start_seg + num_tokens}}}]"
                smiles = ".".join(safe_segs)
                try:
                    new_molecules = inference_fn(smiles=smiles, invoke_url=invoke_url, **inference_cfg)
                    similar_molecules.extend([new_mol["smiles"] for new_mol in new_molecules])
                except Exception as e:
                    print(f"Error: {e}")
                    continue

            partition_results[reference] = similar_molecules
        return partition_results

    def produce_similar_smiles(self, molecules: pd.Series, cfg: GenMolProduceConfig) -> dict[str, list[str]]:
        """Generate similar molecules using ProcessPoolExecutor with N workers."""
        total_api_instances = len(cfg.invoke_urls)
        if total_api_instances == 0:
            raise ValueError("Must pass at least one invocation url")
        if total_api_instances == 1:
            return self._process_partition(molecules, cfg, self.inference, cfg.invoke_urls[0])

        partitions = self.generate_partitions(molecules, cfg)
        if partitions is None:
            raise ValueError("Number of partitions must be greater than 0")

        invoke_urls = list(cfg.invoke_urls)
        partition_dicts: list[dict[str, list[str]]] = []
        with ProcessPoolExecutor(max_workers=total_api_instances) as executor:
            futures = [
                executor.submit(self._process_partition, part, cfg, self.inference, url)
                for part, url in zip(partitions, invoke_urls, strict=True)
            ]
            for future in tqdm(futures, desc="Generating similar SMILES"):
                partition_dicts.append(future.result())  # Aggregate results

        combined_dict = {k: v for d in partition_dicts for k, v in d.items()}
        return combined_dict

    def inference(self, **params):
        invoke_url = params.pop("invoke_url", self.invoke_url)
        task = GenMolGenerator.__default_params__.copy()
        task.update(params)

        if self.verbose:
            print("TASK:", str(task))

        json_data = {k: str(v) for k, v in task.items()}

        try:
            response = self.session.post(invoke_url, headers=self.headers, json=json_data)
            response.raise_for_status()

            output = response.json()
            assert output["status"] == "success"
            return output["molecules"]
        except Exception as e:
            if self.verbose:
                print(f"Request failed or returned an error: {e}")
            return []

    def generate_partitions(self, molecules: pd.Series, cfg: GenMolProduceConfig) -> list | None:
        """Generate partitions of molecules for parallel processing.
        Each partition (GPU) might have multiple API instances associated with it.
        This generates a partition for each GPU and its sub-partitioned for each API instance.
        Each element in `api_instances_per_partition` represents the number of API instances
        associated with that GPU.
        """
        total_molecules = len(molecules)
        num_partitions = len(cfg.api_instances_per_partition)

        if num_partitions == 0:
            return None

        if len(cfg.partition_fractions) != num_partitions:
            raise ValueError(
                f"Number of partition fractions ({len(cfg.partition_fractions)}) "
                f"must match number of partitions ({num_partitions})"
            )

        if not np.isclose(sum(cfg.partition_fractions), 1.0, atol=1e-5):
            raise ValueError(f"Partition fractions must sum to 1.0, got {sum(cfg.partition_fractions)}")

        total_api_instances = sum(cfg.api_instances_per_partition)
        if len(cfg.invoke_urls) != total_api_instances:
            raise ValueError(
                f"Number of invocation URLs ({len(cfg.invoke_urls)}) "
                f"must match total API instances ({total_api_instances})"
            )

        if num_partitions == 1:
            return list(np.array_split(molecules, cfg.api_instances_per_partition[0]))

        partitions: list = []
        start_idx = 0
        for i in range(num_partitions):
            # Calculate segment size based on total molecules and specific fraction
            # Use exact calculation for all but last to avoid rounding drift
            if i == num_partitions - 1:
                end_idx = total_molecules
            else:
                segment_size = int(total_molecules * cfg.partition_fractions[i])
                end_idx = start_idx + segment_size

            # Split this segment into the specified number of sub-partitions
            segment = molecules[start_idx:end_idx]
            sub_partitions = np.array_split(segment, cfg.api_instances_per_partition[i])
            partitions.extend(sub_partitions)

            start_idx = end_idx
        return partitions
