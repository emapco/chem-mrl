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

                new_molecules = inference_fn(smiles=smiles, invoke_url=invoke_url, **inference_cfg)
                similar_molecules.extend([_["smiles"] for _ in new_molecules])

            partition_results[reference] = similar_molecules
        return partition_results

    def produce_similar_smiles(
        self, molecules: pd.Series, cfg: GenMolProduceConfig
    ) -> dict[str, list[str]]:
        """Generate similar molecules using ProcessPoolExecutor with N workers."""
        num_workers = cfg.num_workers or 1
        if num_workers == 1:
            return self._process_partition(molecules, cfg, self.inference, cfg.invoke_urls[0])

        partitions = np.array_split(molecules, cfg.num_workers)  # Split into N partitions
        invoke_urls = list(cfg.invoke_urls)
        assert len(invoke_urls) == cfg.num_workers, (
            "Number of invoke_urls must match number of workers"
        )

        partition_dicts: list[dict[str, list[str]]] = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._process_partition, part, cfg, self.inference, url)
                for part, url in zip(partitions, invoke_urls)
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

        response = self.session.post(invoke_url, headers=self.headers, json=json_data)
        response.raise_for_status()

        output = response.json()
        assert output["status"] == "success"
        return output["molecules"]
